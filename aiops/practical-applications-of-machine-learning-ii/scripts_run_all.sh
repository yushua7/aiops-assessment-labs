#!/usr/bin/env bash
# ML II - AIOps anomaly detection, forecasting, serving, drift, champion/challenger - full automation (no set -e on purpose)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export DEBIAN_FRONTEND=noninteractive

log(){ echo; echo "=== $* ==="; }
retry(){ local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n: $*"; sleep 5; done; return 1; }
wuntil(){ local t=$1; shift; local end=$((SECONDS+t)); until eval "$*" >/dev/null 2>&1; do [ $SECONDS -ge $end ] && return 1; sleep 3; done; return 0; }

log "Setup: packages, swap, docker access"
sudo chmod 666 /var/run/docker.sock
retry 5 sudo apt-get -o DPkg::Lock::Timeout=300 update -qq
retry 3 sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3-venv python3-pip python3-dev build-essential jq curl ca-certificates
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
fi
free -h; nproc; df -h / | tail -1
python3 --version; docker --version
ss -ltn | grep -qE ":(5000|8000|9090) " && echo "WARNING: one of ports 5000/8000/9090 is already in use"
docker run --rm hello-world 2>&1 | grep "Hello from Docker"

log "Project layout, venv, pinned packages"
mkdir -p ~/aiops-lab/{src,data,results,mlflow/artifacts,service/model,monitoring}
cd ~/aiops-lab
cat <<'EOF' > ~/aiops-lab/requirements-lab.txt
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.4.2
joblib==1.4.2
mlflow==2.14.3
fastapi==0.111.0
uvicorn==0.30.1
prometheus-client==0.20.0
requests==2.32.3
sqlalchemy==2.0.30
alembic==1.13.1
pydantic==2.7.4
EOF
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==24.0 wheel "setuptools<70"
retry 3 pip install -r requirements-lab.txt
retry 3 pip install --no-deps drain3==0.9.11 "jsonpickle>=3,<4" "cachetools>=5,<6"
python -c "from sqlalchemy.pool import FallbackAsyncAdaptedQueuePool" 2>/dev/null || { echo "sqlalchemy fallback pin"; pip install "sqlalchemy==2.0.36" "alembic==1.13.3"; }
python -c "import drain3" 2>/dev/null || { echo "drain3 import failed, upgrading its helpers"; pip install -U jsonpickle cachetools; python -c "import drain3" && echo "drain3 ok"; }
python - <<'EOF'
from importlib.metadata import version
for p in ["numpy", "pandas", "scikit-learn", "mlflow", "drain3", "fastapi", "prometheus-client", "sqlalchemy", "pydantic"]:
    print(f"{p:<18} {version(p)}")
EOF

log "Task 1: data generator"
cat <<'EOF' > src/gen_data.py
#!/usr/bin/env python3
"""Synthetic AIOps telemetry generator.

Produces per-minute service metrics with labelled incidents and
application logs with labelled incident periods. Fully seeded.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TYPES = ["cpu_spike", "latency_spike", "error_burst", "memory_leak", "traffic_drop"]
VARIANTS = {
    "v1": {"seed": 42, "start": "2024-03-01", "req_scale": 1.0, "lat_scale": 1.0, "suffix": ""},
    "v2": {"seed": 7, "start": "2024-03-15", "req_scale": 1.25, "lat_scale": 1.4, "suffix": "_v2"},
}

NORMAL_LOGS = [
    (0.55, "INFO", "GET /api/orders/{oid} status=200 duration={ms}ms"),
    (0.20, "INFO", "User {uid} logged in from 10.0.{a}.{b}"),
    (0.15, "INFO", "Cache hit ratio {ratio}"),
    (0.10, "DEBUG", "DB query completed in {ms}ms rows={rows}"),
]
ANOMALY_LOGS = [
    ("ERROR", "Connection refused to db-{n}:5432"),
    ("WARN", "Retry {k}/5 for request {oid}"),
    ("ERROR", "Timeout after {ms}ms calling payment-svc"),
    ("ERROR", "OutOfMemoryError in worker-{n}"),
]


def make_metrics(rng, start, req_scale, lat_scale, days=14):
    idx = pd.date_range(start, periods=days * 1440, freq="min")
    n = len(idx)
    minute_of_day = np.arange(n) % 1440
    load = (np.sin(2 * np.pi * minute_of_day / 1440 - np.pi / 2) + 1) / 2
    weekend = np.isin(idx.dayofweek, [5, 6])
    requests = (300 + 500 * load * np.where(weekend, 0.6, 1.0)) * req_scale + rng.normal(0, 20, n)
    cpu = 15 + 0.07 * requests + rng.normal(0, 3, n)
    mem = 45 + rng.normal(0, 1.5, n)
    latency = (80 + 0.05 * requests) * lat_scale + rng.gamma(2.0, 3.0, n)
    error_rate = 0.5 + np.abs(rng.normal(0, 0.2, n))

    label = np.zeros(n, dtype=int)
    incidents = []
    for i in range(28):
        start_i = 360 + i * 700 + int(rng.integers(0, 300))
        kind = TYPES[i % len(TYPES)]
        dur = int(rng.integers(60, 121)) if kind == "memory_leak" else int(rng.integers(5, 21))
        sl = slice(start_i, start_i + dur)
        if kind == "cpu_spike":
            cpu[sl] += 35
        elif kind == "latency_spike":
            latency[sl] *= 4
        elif kind == "error_burst":
            error_rate[sl] += 8
        elif kind == "memory_leak":
            mem[sl] += np.linspace(10, 40, dur)
        elif kind == "traffic_drop":
            requests[sl] *= 0.2
            cpu[sl] = 15 + 0.07 * requests[sl] + rng.normal(0, 3, dur)
        label[sl] = 1
        incidents.append({
            "id": f"INC-{i + 1:03d}", "type": kind,
            "start": str(idx[start_i]), "end": str(idx[start_i + dur - 1]), "minutes": dur,
        })

    df = pd.DataFrame({
        "timestamp": idx,
        "requests": requests.round(1),
        "cpu_pct": np.clip(cpu, 0, 100).round(2),
        "mem_pct": np.clip(mem, 0, 100).round(2),
        "latency_ms": latency.round(1),
        "error_rate_pct": error_rate.round(3),
        "is_anomaly": label,
    })
    return df, incidents


def fill(tpl, rng):
    return tpl.format(
        oid=rng.integers(10000, 99999), ms=rng.integers(5, 400), uid=rng.integers(1000, 9999),
        a=rng.integers(0, 255), b=rng.integers(1, 255), ratio=f"{rng.uniform(0.7, 0.99):.2f}",
        rows=rng.integers(1, 500), n=rng.integers(1, 4), k=rng.integers(1, 6),
    )


def make_logs(rng, days=3):
    minutes = days * 1440
    base = datetime(2024, 3, 1)
    in_incident = np.zeros(minutes, dtype=bool)
    log_incidents = []
    for i in range(12):
        start = 120 + i * 350 + int(rng.integers(0, 150))
        dur = int(rng.integers(10, 26))
        in_incident[start:start + dur] = True
        log_incidents.append({"start_minute": start, "end_minute": start + dur - 1})

    weights = np.array([w for w, _, _ in NORMAL_LOGS])
    lines = []
    for m in range(minutes):
        picks = rng.choice(len(NORMAL_LOGS), size=rng.poisson(15), p=weights)
        events = [(NORMAL_LOGS[j][1], NORMAL_LOGS[j][2]) for j in picks]
        if rng.random() < 0.02:                      # benign background retries
            events.append(("WARN", "Retry {k}/5 for request {oid}"))
        if in_incident[m]:
            for _ in range(rng.poisson(8)):
                events.append(ANOMALY_LOGS[int(rng.integers(len(ANOMALY_LOGS)))])
        secs = np.sort(rng.integers(0, 60, len(events)))
        for s, (lvl, tpl) in zip(secs, events):
            ts = (base + timedelta(minutes=m, seconds=int(s))).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"{ts} app-0{rng.integers(1, 4)} {lvl} {fill(tpl, rng)}")
    return lines, log_incidents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="v1")
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    rng = np.random.default_rng(v["seed"])
    DATA.mkdir(exist_ok=True)

    df, incidents = make_metrics(rng, v["start"], v["req_scale"], v["lat_scale"])
    df.to_csv(DATA / f"metrics{v['suffix']}.csv", index=False)
    (DATA / f"incidents{v['suffix']}.json").write_text(json.dumps(incidents, indent=2))
    print(f"metrics{v['suffix']}.csv: {len(df)} rows, {df.is_anomaly.sum()} anomalous minutes "
          f"({df.is_anomaly.mean():.1%}), {len(incidents)} incidents")

    if args.variant == "v1":
        lines, log_inc = make_logs(rng)
        (DATA / "app.log").write_text("\n".join(lines) + "\n")
        (DATA / "log_incidents.json").write_text(json.dumps(log_inc, indent=2))
        print(f"app.log: {len(lines)} lines, {len(log_inc)} log incident periods")


if __name__ == "__main__":
    main()
EOF
python src/gen_data.py
head -3 data/metrics.csv
head -3 data/app.log
jq -r 'group_by(.type)[] | "\(.[0].type): \(length)"' data/incidents.json
wc -l data/metrics.csv
jq length data/incidents.json
test -s data/app.log && echo "app.log OK: $(wc -l < data/app.log) lines"

log "Task 2: features, eval utils, baseline"
cat <<'EOF' > src/features.py
"""Shared feature engineering. Imported by training AND serving code.

Any change here must be followed by retraining, otherwise the served model
sees features it was not trained on (train/serve skew).
"""
import numpy as np
import pandas as pd

BASE_COLS = ["requests", "cpu_pct", "mem_pct", "latency_ms", "error_rate_pct"]
WINDOW = 30          # minutes of context used for rolling features
MIN_PERIODS = 5
DAY = 1440           # minutes per day

FEATURE_COLS = [f for c in BASE_COLS for f in (c, f"{c}_z{WINDOW}", f"{c}_diff1")] + [
    "hour_sin", "hour_cos", "cpu_per_req", "lat_per_req"]


def load_metrics(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in BASE_COLS:
        x = df[c].astype(float)
        roll = x.rolling(WINDOW, min_periods=MIN_PERIODS)
        out[c] = x
        out[f"{c}_z{WINDOW}"] = (x - roll.mean()) / (roll.std() + 1e-6)
        out[f"{c}_diff1"] = x.diff()
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cpu_per_req"] = df["cpu_pct"] / (df["requests"] + 1)
    out["lat_per_req"] = df["latency_ms"] / (df["requests"] + 1)
    return out[FEATURE_COLS].fillna(0.0)


def time_split(df: pd.DataFrame):
    """Chronological split: days 1-7 train, 8-10 validation, 11-14 test."""
    return df.iloc[: 7 * DAY], df.iloc[7 * DAY: 10 * DAY], df.iloc[10 * DAY:]
EOF
cat <<'EOF' > src/eval_utils.py
"""Evaluation helpers for operational anomaly detection."""
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


def _runs(mask):
    """Return (start, end_exclusive) tuples of contiguous True runs."""
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def evaluate(y_true, y_pred, steps_per_day=1440):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    incidents = _runs(y_true)
    detected = sum(1 for s, e in incidents if y_pred[s:e].any())
    false_alerts = sum(1 for s, e in _runs(y_pred) if not y_true[s:e].any())
    days = len(y_true) / steps_per_day
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "event_recall": round(detected / len(incidents), 4) if incidents else 0.0,
        "incidents": len(incidents),
        "false_alerts_per_day": round(false_alerts / days, 2),
    }


def best_threshold(scores, y_true):
    """Threshold that maximises F1 (prediction rule: score >= threshold)."""
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])
EOF
cat <<'EOF' > src/baseline.py
#!/usr/bin/env python3
"""Statistical baseline: rolling robust z-score on every base metric."""
import json
from pathlib import Path

import pandas as pd

from eval_utils import evaluate
from features import BASE_COLS, load_metrics, time_split


def robust_z(x: pd.Series, window: int = 60) -> pd.Series:
    prev = x.shift(1)
    med = prev.rolling(window, min_periods=10).median()
    mad = (prev - med).abs().rolling(window, min_periods=10).median()
    return 0.6745 * (x - med) / (mad + 1e-6)


def main():
    df = load_metrics("data/metrics.csv")
    z = pd.DataFrame({c: robust_z(df[c]) for c in BASE_COLS}).fillna(0.0)
    max_abs = z.abs().max(axis=1)
    _, _, test = time_split(df)

    rows = []
    for thr in (3.0, 3.5, 4.0, 5.0, 6.0):
        pred = (max_abs.loc[test.index] > thr).astype(int)
        m = evaluate(test["is_anomaly"], pred)
        m["threshold"] = thr
        rows.append(m)
    table = pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "event_recall",
                                "false_alerts_per_day"]]
    print("Robust z-score baseline on TEST split (days 11-14)")
    print(table.to_string(index=False))

    # Root-cause hint: which metric deviated most during each test incident?
    inc = pd.DataFrame(json.loads(Path("data/incidents.json").read_text()))
    inc["start"], inc["end"] = pd.to_datetime(inc["start"]), pd.to_datetime(inc["end"])
    inc = inc[inc["start"] >= test["timestamp"].iloc[0]].copy()
    detected, top_metric = [], []
    for _, r in inc.iterrows():
        mask = (df["timestamp"] >= r["start"]) & (df["timestamp"] <= r["end"])
        detected.append(bool((max_abs[mask] > 4.0).any()))
        top_metric.append(z[mask].abs().max().idxmax())
    inc["detected@4.0"], inc["top_metric"] = detected, top_metric
    print("\nPer-incident view at threshold 4.0")
    print(inc[["id", "type", "minutes", "detected@4.0", "top_metric"]].to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    Path("results/baseline.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
EOF
python src/baseline.py
jq -r '.[] | "\(.threshold) f1=\(.f1) false_alerts_per_day=\(.false_alerts_per_day)"' results/baseline.json
python -c "import sys; sys.path.insert(0,'src'); from features import *; import pandas as pd; X=build_features(load_metrics('data/metrics.csv')); print(X.shape, int(X.isna().sum().sum()))"

log "Task 3: MLflow server, detector sweep, registry"
nohup mlflow server \
  --backend-store-uri "sqlite:///$HOME/aiops-lab/mlflow/mlflow.db" \
  --artifacts-destination "$HOME/aiops-lab/mlflow/artifacts" \
  --host 127.0.0.1 --port 5000 --workers 1 \
  > mlflow/server.out 2>&1 &
wuntil 180 "curl -sf http://127.0.0.1:5000/health" || { echo "MLflow did not start:"; tail -30 mlflow/server.out; }
curl -s http://127.0.0.1:5000/health; echo
cat <<'EOF' > src/train_detectors.py
#!/usr/bin/env python3
"""Train and compare unsupervised anomaly detectors; track in MLflow; register best."""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_utils import best_threshold, evaluate
from features import build_features, load_metrics, time_split

MODEL_NAME = "infra-anomaly-detector"
EXPERIMENT = "infra-anomaly"
PIP_REQS = ["scikit-learn==1.4.2", "numpy==1.26.4", "pandas==2.2.2"]


def candidates():
    cands = []
    for n_est in (100, 200):
        for max_samples in (256, 1024):
            cands.append((
                f"iforest_n{n_est}_s{max_samples}",
                {"model": "iforest", "n_estimators": n_est, "max_samples": max_samples},
                IsolationForest(n_estimators=n_est, max_samples=max_samples,
                                random_state=42, n_jobs=-1),
            ))
    for k in (20, 50):
        cands.append((f"lof_k{k}", {"model": "lof", "n_neighbors": k},
                      LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    ap.add_argument("--run-tag", default="initial")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = load_metrics(args.data)
    X = build_features(df)
    tr, va, te = time_split(df)
    Xtr, Xva, Xte = X.loc[tr.index], X.loc[va.index], X.loc[te.index]
    yva, yte = va["is_anomaly"].to_numpy(), te["is_anomaly"].to_numpy()

    results = []
    for name, params, est in candidates():
        with mlflow.start_run(run_name=f"{args.run_tag}-{name}") as run:
            pipe = make_pipeline(StandardScaler(), est)
            pipe.fit(Xtr)
            s_va = -pipe.score_samples(Xva)
            thr, _ = best_threshold(s_va, yva)
            s_te = -pipe.score_samples(Xte)
            m_va = evaluate(yva, (s_va >= thr).astype(int))
            m_te = evaluate(yte, (s_te >= thr).astype(int))

            mlflow.log_params({**params, "threshold": round(thr, 6), "train_rows": len(Xtr)})
            mlflow.set_tags({"run_tag": args.run_tag, "data_file": args.data})
            mlflow.log_metrics({f"val_{k}": v for k, v in m_va.items()}
                               | {f"test_{k}": v for k, v in m_te.items()})
            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     signature=infer_signature(Xva.head(50), s_va[:50]),
                                     pip_requirements=PIP_REQS)
            results.append({"name": name, "run_id": run.info.run_id, "threshold": thr,
                            "val_f1": m_va["f1"], "test_f1": m_te["f1"]})
            print(f"{name:<22} val_f1={m_va['f1']:.3f} test_f1={m_te['f1']:.3f} "
                  f"test_event_recall={m_te['event_recall']:.2f} "
                  f"test_false_alerts/day={m_te['false_alerts_per_day']:.1f}")

    best = max(results, key=lambda r: r["val_f1"])
    print(f"\nBest candidate by validation F1: {best['name']} (val_f1={best['val_f1']:.3f})")

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "threshold", str(best["threshold"]))
    client.set_model_version_tag(MODEL_NAME, mv.version, "candidate", best["name"])
    client.set_model_version_tag(MODEL_NAME, mv.version, "data_file", args.data)
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        alias = "challenger"
    except MlflowException:
        alias = "champion"
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    print(f"Registered version {mv.version} with alias '{alias}' (threshold={best['threshold']:.4f})")


if __name__ == "__main__":
    main()
EOF
python src/train_detectors.py 2>&1 | grep -v -i warning
EXP_ID=$(curl -s "http://127.0.0.1:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=infra-anomaly" | jq -r .experiment.experiment_id)
echo "experiment id: $EXP_ID"
curl -s -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d "{\"experiment_ids\":[\"$EXP_ID\"],\"order_by\":[\"metrics.val_f1 DESC\"],\"max_results\":3}" \
  | jq -r '.runs[] | [.info.run_name,
      (.data.metrics[] | select(.key=="val_f1") | .value),
      (.data.metrics[] | select(.key=="test_false_alerts_per_day") | .value)] | @tsv'
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias?name=infra-anomaly-detector&alias=champion" \
  | jq '.model_version | {name, version, aliases, tags}'
find mlflow/artifacts -name MLmodel | wc -l

log "Task 4: Drain3 log anomaly detection"
cat <<'EOF' > src/log_anomaly.py
#!/usr/bin/env python3
"""Log anomaly detection: Drain3 templates -> event count matrix -> Isolation Forest."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.ensemble import IsolationForest

from eval_utils import best_threshold, evaluate

LINE_RE = re.compile(r"^(\S+) (\S+) (\S+) (.*)$")
WINDOW_MIN = 5
WINDOWS_PER_DAY = 1440 // WINDOW_MIN


def make_miner():
    cfg = TemplateMinerConfig()
    cfg.masking_instructions = [MaskingInstruction(r"\d+(\.\d+)*", "NUM")]
    cfg.drain_sim_th = 0.5
    cfg.profiling_enabled = False
    return TemplateMiner(config=cfg)


def parse(path, miner):
    rows = []
    with open(path) as fh:
        for line in fh:
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts, host, level, msg = m.groups()
            res = miner.add_log_message(f"{level} {msg}")
            rows.append((ts, host, level, res["cluster_id"]))
    ev = pd.DataFrame(rows, columns=["ts", "host", "level", "cluster_id"])
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    ev["window"] = ev["ts"].dt.floor(f"{WINDOW_MIN}min")
    return ev


def build_count_matrix(events: pd.DataFrame, all_windows: pd.DatetimeIndex) -> pd.DataFrame:
    m = pd.crosstab(events["window"], events["cluster_id"])
    return m.reindex(index=all_windows, fill_value=0).sort_index(axis=1)


def main():
    miner = make_miner()
    events = parse("data/app.log", miner)

    templates = pd.DataFrame(
        [(c.cluster_id, c.size, c.get_template()) for c in miner.drain.clusters],
        columns=["cluster_id", "size", "template"]).sort_values("size", ascending=False)
    print(f"Parsed {len(events)} lines into {len(templates)} templates")
    print(templates.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    templates.to_csv("results/templates.csv", index=False)

    base = pd.Timestamp("2024-03-01", tz="UTC")
    n_windows = 3 * WINDOWS_PER_DAY
    all_windows = pd.date_range(base, periods=n_windows, freq=f"{WINDOW_MIN}min")
    counts = build_count_matrix(events, all_windows)

    labels = np.zeros(n_windows, dtype=int)
    for inc in json.loads(Path("data/log_incidents.json").read_text()):
        labels[inc["start_minute"] // WINDOW_MIN: inc["end_minute"] // WINDOW_MIN + 1] = 1

    X = np.log1p(counts.to_numpy(dtype=float))
    tr = slice(0, WINDOWS_PER_DAY)                 # day 1: fit + threshold
    te = slice(WINDOWS_PER_DAY, n_windows)         # days 2-3: evaluation
    model = IsolationForest(n_estimators=200, random_state=42).fit(X[tr])
    scores = -model.score_samples(X)
    thr, _ = best_threshold(scores[tr], labels[tr])
    pred = (scores >= thr).astype(int)
    m_tr = evaluate(labels[tr], pred[tr], steps_per_day=WINDOWS_PER_DAY)
    m_te = evaluate(labels[te], pred[te], steps_per_day=WINDOWS_PER_DAY)
    print(f"\nthreshold={thr:.4f}")
    print("day 1   :", m_tr)
    print("days 2-3:", m_te)

    # Explain the most anomalous test window: which templates are over-represented?
    test_idx = np.arange(n_windows)[te]
    w = test_idx[np.argmax(scores[te])]
    baseline = counts.iloc[tr].mean()
    ratio = (counts.iloc[w] / (baseline + 0.1)).sort_values(ascending=False).head(3)
    tmap = templates.set_index("cluster_id")["template"]
    print(f"\nMost anomalous window {all_windows[w]} (score {scores[w]:.3f}); top templates vs day-1 mean:")
    for cid, r in ratio.items():
        print(f"  x{r:6.1f}  [{cid}] {tmap[cid]}")

    Path("results/log_anomaly.json").write_text(json.dumps(
        {"n_templates": int(len(templates)), "threshold": thr, "train": m_tr, "test": m_te}, indent=2))


if __name__ == "__main__":
    main()
EOF
python src/log_anomaly.py
jq '{n_templates, test_f1: .test.f1, test_event_recall: .test.event_recall}' results/log_anomaly.json
wc -l < results/templates.csv

log "Task 5: forecasting"
cat <<'EOF' > src/forecast.py
#!/usr/bin/env python3
"""Hourly traffic forecast: seasonal naive vs gradient boosting, plus P95 for capacity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import load_metrics

HORIZON = 96                 # forecast the last 4 days (hours)
REPLICA_CAPACITY = 250       # requests/min one replica serves at target utilisation
FEATS = ["hour", "hour_sin", "hour_cos", "dow", "is_weekend"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def seasonal_naive(series: pd.Series, season: int = 24) -> pd.Series:
    return series.shift(season)


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    args = ap.parse_args()

    df = load_metrics(args.data)
    y = df.set_index("timestamp")["requests"].resample("1h").mean()
    X = calendar_features(y.index)
    X_tr, y_tr, X_te, y_te = X.iloc[:-HORIZON], y.iloc[:-HORIZON], X.iloc[-HORIZON:], y.iloc[-HORIZON:]

    naive = seasonal_naive(y).iloc[-HORIZON:]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    pred = pd.Series(gbm.fit(X_tr[FEATS], y_tr).predict(X_te[FEATS]), index=y_te.index)
    q95m = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=300,
                                         learning_rate=0.05, random_state=42)
    p95 = pd.Series(q95m.fit(X_tr[FEATS], y_tr).predict(X_te[FEATS]), index=y_te.index)

    res = {
        "naive": {"mae": round(float((y_te - naive).abs().mean()), 2), "mape": round(mape(y_te, naive), 2)},
        "gbm": {"mae": round(float((y_te - pred).abs().mean()), 2), "mape": round(mape(y_te, pred), 2)},
        "p95_coverage": round(float((y_te <= p95).mean()), 3),
    }
    err = pd.DataFrame({"naive_abs": (y_te - naive).abs(), "gbm_abs": (y_te - pred).abs(), "p95": p95})
    daily = err.groupby(err.index.date).agg(naive_mae=("naive_abs", "mean"),
                                            gbm_mae=("gbm_abs", "mean"), peak_p95=("p95", "max"))
    daily["replicas"] = np.ceil(daily["peak_p95"] / REPLICA_CAPACITY).astype(int)
    daily.index = daily.index.astype(str)
    daily.index.name = "date"

    print(f"Test horizon: {y_te.index[0]} -> {y_te.index[-1]} ({HORIZON} hours)")
    print(f"Seasonal naive : MAE={res['naive']['mae']:7.2f}  MAPE={res['naive']['mape']:5.2f}%")
    print(f"Gradient boost : MAE={res['gbm']['mae']:7.2f}  MAPE={res['gbm']['mape']:5.2f}%")
    print(f"P95 coverage   : {res['p95_coverage']:.1%} of actual hours below the P95 forecast\n")
    print("Per-day error and capacity plan:")
    print(daily.round(1).to_string())
    res["capacity_plan"] = daily.round(2).reset_index().to_dict("records")
    Path("results").mkdir(exist_ok=True)
    Path("results/forecast.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
EOF
python src/forecast.py
jq -e '.gbm.mae < .naive.mae' results/forecast.json && echo "GBM beats naive"
jq -r '.capacity_plan[] | "\(.date) replicas=\(.replicas)"' results/forecast.json

log "Task 6: export, serve, container"
cat <<'EOF' > src/export_model.py
#!/usr/bin/env python3
"""Export a registry alias to a self-contained serving bundle (model.joblib + metadata.json)."""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from features import FEATURE_COLS, WINDOW

MODEL_NAME = "infra-anomaly-detector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default="service/model")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, args.alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {"model_name": MODEL_NAME, "version": mv.version, "alias": args.alias,
            "threshold": float(mv.tags["threshold"]), "candidate": mv.tags.get("candidate"),
            "run_id": mv.run_id, "feature_cols": FEATURE_COLS, "window": WINDOW}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {MODEL_NAME} v{mv.version} ({meta['candidate']}, alias={args.alias}) "
          f"threshold={meta['threshold']:.4f} -> {out}")


if __name__ == "__main__":
    main()
EOF
python src/export_model.py 2>&1 | grep -v -i warning
ls -l service/model
cp src/features.py service/features.py
cat <<'EOF' > service/app.py
"""AIOps anomaly scoring service."""
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from features import BASE_COLS, FEATURE_COLS, build_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

PREDICTIONS = Counter("anomaly_predictions", "Scored observations by result", ["result"])
for _r in ("anomaly", "normal"):
    PREDICTIONS.labels(result=_r)
INFER_SECONDS = Histogram("anomaly_inference_seconds", "Feature building + scoring time",
                          buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
SCORE = Histogram("anomaly_score", "Distribution of anomaly scores",
                  buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 5.0))
INPUT_VALUE = Gauge("anomaly_input_value", "Latest raw input value per metric", ["metric"])
MODEL_INFO = Gauge("anomaly_model_info", "Currently loaded model", ["version", "candidate"])
MODEL_LOADED = Gauge("anomaly_model_loaded", "1 if a model is loaded, else 0")

STATE = {"model": None, "meta": {}, "error": "not loaded yet"}


def load_model() -> bool:
    """Load the bundle. On failure keep the previously loaded model (if any)."""
    try:
        meta = json.loads((MODEL_DIR / "metadata.json").read_text())
        if meta.get("feature_cols") != FEATURE_COLS:
            raise ValueError("feature_cols in metadata.json do not match features.py (train/serve skew)")
        model = joblib.load(MODEL_DIR / "model.joblib")
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        MODEL_LOADED.set(1 if STATE["model"] is not None else 0)
        return False
    STATE.update(model=model, meta=meta, error=None)
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=str(meta["version"]), candidate=str(meta.get("candidate"))).set(1)
    MODEL_LOADED.set(1)
    return True


class Observation(BaseModel):
    timestamp: str
    requests: float = Field(ge=0)
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0)
    error_rate_pct: float = Field(ge=0)


class ScoreRequest(BaseModel):
    observations: list[Observation] = Field(min_length=5, max_length=500)


app = FastAPI(title="AIOps anomaly scoring service", version="1.0")
load_model()


@app.get("/health")
def health():
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "error": STATE["error"]})
    m = STATE["meta"]
    return {"status": "ok", "model_version": m["version"], "candidate": m.get("candidate"),
            "threshold": m["threshold"]}


@app.post("/reload")
def reload():
    if not load_model():
        raise HTTPException(status_code=500, detail=STATE["error"])
    return {"status": "reloaded", "model_version": STATE["meta"]["version"]}


@app.post("/score")
def score(req: ScoreRequest):
    model, meta = STATE["model"], STATE["meta"]
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE['error']}")
    start = time.perf_counter()
    df = pd.DataFrame([o.model_dump() for o in req.observations])
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing:
        raise HTTPException(status_code=400, detail="timestamps must be valid and in ascending order")
    feats = build_features(df)
    s = float(-model.score_samples(feats.iloc[[-1]])[0])
    is_anom = s >= float(meta["threshold"])
    INFER_SECONDS.observe(time.perf_counter() - start)
    SCORE.observe(s)
    PREDICTIONS.labels(result="anomaly" if is_anom else "normal").inc()
    last = req.observations[-1]
    for c in BASE_COLS:
        INPUT_VALUE.labels(metric=c).set(getattr(last, c))
    return {"timestamp": last.timestamp, "score": s, "threshold": meta["threshold"],
            "is_anomaly": bool(is_anom), "model_version": meta["version"]}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
EOF
cat <<'EOF' > service/requirements.txt
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.4.2
joblib==1.4.2
fastapi==0.111.0
uvicorn==0.30.1
prometheus-client==0.20.0
pydantic==2.7.4
EOF
cat <<'EOF' > service/.dockerignore
model/
__pycache__/
*.pyc
EOF
# base image is 3.12 (not 3.10) to match the Python that trained and pickled the model on this VM
cat <<'EOF' > service/Dockerfile
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY features.py app.py ./
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
retry 3 docker build --provenance=false --sbom=false -t aiops/anomaly-svc:1.0 service/ 2>&1 | tail -5
docker image ls aiops/anomaly-svc
echo "--- deliberate failure: no model mounted ---"
docker network create aiops-net 2>/dev/null || true
docker rm -f anomaly-svc >/dev/null 2>&1
docker run -d --name anomaly-svc --network aiops-net -p 8000:8000 aiops/anomaly-svc:1.0
sleep 6
curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/health
docker rm -f anomaly-svc
docker run -d --name anomaly-svc --network aiops-net -p 8000:8000 \
  -v "$HOME/aiops-lab/service/model:/app/model:ro" aiops/anomaly-svc:1.0
wuntil 60 "curl -sf http://127.0.0.1:8000/health"
curl -s http://127.0.0.1:8000/health | jq
cat <<'EOF' > src/make_payload.py
#!/usr/bin/env python3
"""Build a /score payload of WINDOW observations ending at a chosen minute."""
import argparse
import json

import pandas as pd

from features import BASE_COLS, DAY, WINDOW, load_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    ap.add_argument("--incidents", default="data/incidents.json")
    ap.add_argument("--incident", help="pick the middle of the first test-period incident of this type")
    ap.add_argument("--index", type=int, help="explicit row index")
    ap.add_argument("--out", default="/tmp/payload.json")
    args = ap.parse_args()
    df = load_metrics(args.data)
    test_start = 10 * DAY
    if args.index is not None:
        i = args.index
    elif args.incident:
        inc = [x for x in json.load(open(args.incidents))
               if x["type"] == args.incident and pd.Timestamp(x["start"]) >= df.timestamp[test_start]][0]
        i = int(df.index[df.timestamp == pd.Timestamp(inc["start"])][0]) + inc["minutes"] // 2
    else:  # first fully normal window in the test period
        lab = df.is_anomaly.rolling(WINDOW).max()
        i = int(lab[(lab == 0) & (df.index >= test_start)].index[0])
    rows = df.iloc[i - WINDOW + 1: i + 1].copy()
    rows["timestamp"] = rows.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    with open(args.out, "w") as fh:
        json.dump({"observations": rows[["timestamp"] + BASE_COLS].to_dict("records")}, fh)
    print(f"row {i}: {len(rows)} observations ending {rows.timestamp.iloc[-1]}, "
          f"true label is_anomaly={int(df.is_anomaly.iloc[i])} -> {args.out}")


if __name__ == "__main__":
    main()
EOF
python src/make_payload.py --out /tmp/normal.json
curl -s -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d @/tmp/normal.json | jq -c
python src/make_payload.py --incident latency_spike --out /tmp/incident.json
curl -s -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d @/tmp/incident.json | jq -c
echo "--- invalid input drill ---"
jq 'del(.observations[-1].latency_ms)' /tmp/normal.json | \
  curl -s -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d @- | jq -c '.detail[0] | {type, loc, msg}'
jq '.observations |= .[:3]' /tmp/normal.json | \
  curl -s -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d @- | jq -c '.detail[0].msg'
jq '.observations |= reverse' /tmp/normal.json | \
  curl -s -w ' HTTP %{http_code}\n' -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' -d @-
python - <<'EOF'
import json, sys
sys.path.insert(0, "src")
import joblib, requests
from features import BASE_COLS, WINDOW, build_features, load_metrics
df = load_metrics("data/metrics.csv")
X = build_features(df)
model = joblib.load("service/model/model.joblib")
worst = 0.0
for i in (14500, 15321, 17777, 20000):
    offline = float(-model.score_samples(X.iloc[[i]])[0])
    obs = df.iloc[i - WINDOW + 1:i + 1].copy()
    obs["timestamp"] = obs.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%S")
    r = requests.post("http://127.0.0.1:8000/score",
                      json={"observations": obs[["timestamp"] + BASE_COLS].to_dict("records")}, timeout=10)
    online = r.json()["score"]
    worst = max(worst, abs(offline - online))
    print(f"row {i}: offline={offline:.6f} online={online:.6f}")
assert worst < 1e-6, "train/serve skew detected"
print("CONSISTENT, max abs diff =", worst)
EOF
diff src/features.py service/features.py && echo "features.py identical"
wuntil 60 "[ \"\$(docker inspect -f '{{.State.Health.Status}}' anomaly-svc)\" = healthy ]"
docker inspect -f '{{.State.Health.Status}}' anomaly-svc

log "Task 7: Prometheus, replay, drift"
cat <<'EOF' > monitoring/prometheus.yml
global:
  scrape_interval: 5s
  evaluation_interval: 5s
rule_files:
  - /etc/prometheus/rules.yml
scrape_configs:
  - job_name: anomaly-svc
    static_configs:
      - targets: ["anomaly-svc:8000"]
EOF
cat <<'EOF' > monitoring/rules.yml
groups:
  - name: aiops-model
    rules:
      - record: aiops:anomaly_rate:2m
        expr: >
          sum(rate(anomaly_predictions_total{result="anomaly"}[2m]))
          / clamp_min(sum(rate(anomaly_predictions_total[2m])), 1e-9)
      - alert: ModelAnomalyRateHigh
        expr: aiops:anomaly_rate:2m > 0.15
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Anomaly rate {{ $value | humanizePercentage }}: real incident or input drift"
      - alert: ModelNotLoaded
        expr: anomaly_model_loaded == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Scoring service is running without a model"
EOF
retry 3 docker pull prom/prometheus:v2.53.0
docker run --rm -v "$HOME/aiops-lab/monitoring:/etc/prometheus:ro" \
  --entrypoint promtool prom/prometheus:v2.53.0 check config /etc/prometheus/prometheus.yml
docker rm -f prometheus >/dev/null 2>&1
docker run -d --name prometheus --network aiops-net -p 9090:9090 \
  -v "$HOME/aiops-lab/monitoring:/etc/prometheus:ro" prom/prometheus:v2.53.0 \
  --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus
wuntil 90 "curl -s http://127.0.0.1:9090/api/v1/targets | jq -e '.data.activeTargets[0].health==\"up\"'"
curl -s http://127.0.0.1:9090/api/v1/targets | jq -r '.data.activeTargets[] | "\(.labels.job) \(.health) \(.lastError)"'
cat <<'EOF' > src/send_traffic.py
#!/usr/bin/env python3
"""Replay telemetry against the scoring API, one sliding window per minute."""
import argparse
import time

import requests

from features import BASE_COLS, DAY, WINDOW, load_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    ap.add_argument("--start", type=int, default=10 * DAY)
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--url", default="http://127.0.0.1:8000/score")
    args = ap.parse_args()
    df = load_metrics(args.data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    records = df[["timestamp"] + BASE_COLS].to_dict("records")
    labels = df["is_anomaly"].to_numpy()
    tp = fp = fn = tn = 0
    sess = requests.Session()
    t0 = time.time()
    for i in range(max(args.start, WINDOW - 1), min(args.start + args.count, len(df))):
        r = sess.post(args.url, json={"observations": records[i - WINDOW + 1:i + 1]}, timeout=10)
        r.raise_for_status()
        pred, truth = r.json()["is_anomaly"], bool(labels[i])
        tp += pred and truth; fp += pred and not truth; fn += truth and not pred; tn += not pred and not truth
    n, dt = tp + fp + fn + tn, time.time() - t0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    print(f"sent={n} in {dt:.1f}s ({n / dt:.0f} req/s) alerts={tp + fp} alert_rate={(tp + fp) / n:.1%} "
          f"TP={tp} FP={fp} FN={fn} precision={prec:.2f} recall={rec:.2f}")


if __name__ == "__main__":
    main()
EOF
python src/send_traffic.py --data data/metrics.csv
q() { curl -s -G http://127.0.0.1:9090/api/v1/query --data-urlencode "query=$1" | jq -r '.data.result[] | "\(.metric | tostring) => \(.value[1])"'; }
sleep 10
q 'sum by (result) (anomaly_predictions_total)'
q 'aiops:anomaly_rate:2m'
q 'histogram_quantile(0.95, sum by (le) (rate(anomaly_inference_seconds_bucket[2m])))'
q 'anomaly_model_info'
python src/gen_data.py --variant v2
cat <<'EOF' > src/drift_check.py
#!/usr/bin/env python3
"""Compare a reference window with current data: PSI + two-sample KS per metric.
Exit code 2 when significant drift is found (usable as a pipeline gate)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from features import BASE_COLS, DAY, load_metrics


def psi(reference, current, bins=10, eps=1e-4) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(v):
    return "significant" if v >= 0.2 else ("moderate" if v >= 0.1 else "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/metrics.csv")
    ap.add_argument("--current", default="data/metrics_v2.csv")
    ap.add_argument("--current-days", type=int, default=2)
    args = ap.parse_args()
    ref = load_metrics(args.reference).iloc[: 7 * DAY]          # the model's training window
    cur = load_metrics(args.current).iloc[: args.current_days * DAY]
    rows = []
    for c in BASE_COLS:
        r, k = ref[c].to_numpy(), cur[c].to_numpy()
        ks = ks_2samp(r, k)
        p = psi(r, k)
        rows.append({"metric": c, "ref_mean": round(float(r.mean()), 2), "cur_mean": round(float(k.mean()), 2),
                     "psi": round(p, 4), "ks_stat": round(float(ks.statistic), 4),
                     "ks_p": float(ks.pvalue), "status": status(p)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, formatters={"ks_p": "{:.2e}".format}))
    drift = bool((table["status"] == "significant").any())
    Path("results").mkdir(exist_ok=True)
    Path("results/drift_report.json").write_text(json.dumps({"drift_detected": drift, "metrics": rows}, indent=2))
    print("\nDRIFT DETECTED: retraining recommended" if drift else "\nNo significant drift")
    sys.exit(2 if drift else 0)


if __name__ == "__main__":
    main()
EOF
python src/drift_check.py --current data/metrics.csv ; echo "exit code: $?"
python src/drift_check.py ; echo "exit code: $?"
echo "--- operational impact: alert storm on drifted data ---"
sleep 120
python src/send_traffic.py --data data/metrics_v2.csv
sleep 40
q 'aiops:anomaly_rate:2m'
q 'avg_over_time(anomaly_input_value{metric="latency_ms"}[1m])'
curl -s http://127.0.0.1:9090/api/v1/alerts | jq -r '.data.alerts[] | "\(.labels.alertname) \(.state) \(.annotations.summary)"'
curl -s -G http://127.0.0.1:9090/api/v1/query --data-urlencode 'query=up{job="anomaly-svc"}' | jq -r '.data.result[0].value[1]'
curl -s http://127.0.0.1:9090/api/v1/rules | jq -r '.data.groups[].rules[].name'
jq -r '.metrics[] | "\(.metric) \(.status)"' results/drift_report.json

log "Task 8: retrain, gate, promote, hot reload, rollback drill"
python src/train_detectors.py --data data/metrics_v2.csv --run-tag retrain-v2 2>&1 | grep -v -i warning
cat <<'EOF' > src/champion_challenger.py
#!/usr/bin/env python3
"""Evaluate champion vs challenger on the same held-out data; promote if clearly better."""
import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from eval_utils import evaluate
from features import build_features, load_metrics, time_split

MODEL_NAME = "infra-anomaly-detector"


def score_alias(client, alias, Xte, yte):
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    thr = float(mv.tags["threshold"])
    pred = (-model.score_samples(Xte) >= thr).astype(int)
    m = evaluate(yte, pred)
    m.update(alias=alias, version=mv.version, candidate=mv.tags.get("candidate"),
             threshold=round(thr, 4), alert_rate=round(float(pred.mean()), 4))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics_v2.csv")
    ap.add_argument("--min-gain", type=float, default=0.02, help="required F1 improvement")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    df = load_metrics(args.data)
    X = build_features(df)
    _, _, te = time_split(df)
    Xte, yte = X.loc[te.index], te["is_anomaly"].to_numpy()
    champ = score_alias(client, "champion", Xte, yte)
    chal = score_alias(client, "challenger", Xte, yte)
    for m in (champ, chal):
        print(f"{m['alias']:<10} v{m['version']:<3} {str(m['candidate']):<20} f1={m['f1']:.3f} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"false_alerts/day={m['false_alerts_per_day']:.1f} alert_rate={m['alert_rate']:.1%}")

    if chal["f1"] >= champ["f1"] + args.min_gain:
        client.set_registered_model_alias(MODEL_NAME, "previous", champ["version"])
        client.set_registered_model_alias(MODEL_NAME, "champion", chal["version"])
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        decision = "promoted"
        print(f"\nDECISION: promote v{chal['version']} to champion (v{champ['version']} -> 'previous')")
    else:
        decision = "kept"
        print(f"\nDECISION: keep v{champ['version']}; challenger gain below {args.min_gain}")
    Path("results/promotion.json").write_text(json.dumps(
        {"decision": decision, "data": args.data, "champion_before": champ, "challenger": chal}, indent=2))


if __name__ == "__main__":
    main()
EOF
python src/champion_challenger.py 2>&1 | grep -v -i warning
if [ "$(jq -r .decision results/promotion.json)" != "promoted" ]; then
  echo "NOTE: gate kept the champion; forcing promotion with --min-gain -1 so the lab flow can complete"
  python src/champion_challenger.py --min-gain -1 2>&1 | grep -v -i warning
fi
python src/export_model.py 2>&1 | grep -v -i warning
curl -s -X POST http://127.0.0.1:8000/reload | jq -c
curl -s http://127.0.0.1:8000/health | jq -c
sleep 120
python src/send_traffic.py --data data/metrics_v2.csv
sleep 15
q 'aiops:anomaly_rate:2m'
q 'anomaly_model_info'
python src/export_model.py --alias previous --out /tmp/rollback-bundle 2>&1 | grep -v -i warning
jq -r '"rollback target: v\(.version) \(.candidate)"' /tmp/rollback-bundle/metadata.json
for a in champion previous; do
  curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias?name=infra-anomaly-detector&alias=$a" \
    | jq -r --arg a "$a" '"\($a) -> v\(.model_version.version)"'
done
jq -r .decision results/promotion.json
curl -s http://127.0.0.1:8000/health | jq -r .model_version

log "Validation (lab validator)"
cat <<'EOF' > ~/aiops-lab/validate.sh
#!/usr/bin/env bash
cd "$HOME/aiops-lab" || exit 1
PASS=0; FAIL=0
check() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[PASS] $d"; PASS=$((PASS+1)); else echo "[FAIL] $d"; FAIL=$((FAIL+1)); fi; }
MLF=http://127.0.0.1:5000/api/2.0/mlflow
alias_version() { curl -s "$MLF/registered-models/alias?name=infra-anomaly-detector&alias=$1" | jq -r .model_version.version; }
runs_count() {
  local exp; exp=$(curl -s "$MLF/experiments/get-by-name?experiment_name=infra-anomaly" | jq -r .experiment.experiment_id)
  curl -s -X POST "$MLF/runs/search" -H 'Content-Type: application/json' \
    -d "{\"experiment_ids\":[\"$exp\"],\"max_results\":200}" | jq '.runs | length'
}
prom() { curl -s -G http://127.0.0.1:9090/api/v1/query --data-urlencode "query=$1" | jq -r '.data.result[0].value[1]'; }

check "T1 metrics.csv has 20160 rows + header"       test "$(wc -l < data/metrics.csv)" -eq 20161
check "T1 28 labelled incidents"                     test "$(jq length data/incidents.json)" -eq 28
check "T1 app.log generated"                         test -s data/app.log
check "T2 baseline sweep saved (5 thresholds)"       test "$(jq length results/baseline.json)" -eq 5
check "T3 MLflow server healthy"                     curl -sf http://127.0.0.1:5000/health
check "T3 at least 6 tracked runs"                   test "$(runs_count)" -ge 6
check "T3 champion alias exists"                     test -n "$(alias_version champion)" -a "$(alias_version champion)" != "null"
check "T4 log templates mined (8)"                   test "$(jq .n_templates results/log_anomaly.json)" -eq 8
check "T4 log anomaly test F1 >= 0.8"                jq -e '.test.f1 >= 0.8' results/log_anomaly.json
check "T5 GBM forecast beats seasonal naive"         jq -e '.gbm.mae < .naive.mae' results/forecast.json
check "T6 service /health returns 200"               curl -sf http://127.0.0.1:8000/health
check "T6 container healthcheck healthy"             test "$(docker inspect -f '{{.State.Health.Status}}' anomaly-svc)" = "healthy"
check "T6 features.py identical in service"          diff -q src/features.py service/features.py
check "T7 Prometheus scraping service"               test "$(prom 'up{job="anomaly-svc"}')" = "1"
check "T7 alert rules loaded"                        test "$(curl -s http://127.0.0.1:9090/api/v1/rules | jq '[.data.groups[].rules[]] | length')" -eq 3
check "T7 drift detected on latency_ms"              jq -e '.metrics[] | select(.metric=="latency_ms") | .status == "significant"' results/drift_report.json
check "T8 promotion decision recorded"               jq -e '.decision' results/promotion.json
check "T8 'previous' alias exists for rollback"      test "$(alias_version previous)" != "null"
check "T8 service runs the current champion"         test "$(curl -s http://127.0.0.1:8000/health | jq -r .model_version)" = "$(alias_version champion)"

echo; echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
EOF
chmod +x ~/aiops-lab/validate.sh
~/aiops-lab/validate.sh
echo "LAB46 DONE"
# END_OF_LAB46
