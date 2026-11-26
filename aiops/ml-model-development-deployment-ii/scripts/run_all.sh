#!/usr/bin/env bash
# ML Model Development and Deployment II - Incident Predictor (MLflow, FastAPI, Docker, kind, drift, gated promotion) - full automation (no set -e on purpose)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export DEBIAN_FRONTEND=noninteractive

log(){ echo; echo "=== $* ==="; }
retry(){ local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n: $*"; sleep 5; done; return 1; }
wuntil(){ local t=$1; shift; local end=$((SECONDS+t)); until eval "$*" >/dev/null 2>&1; do [ $SECONDS -ge $end ] && return 1; sleep 3; done; return 0; }
NODE=aiops-control-plane
# load a local image into the kind node (kind load first, ctr import fallback for Docker 29 digest problems)
loadimg(){
  kind load docker-image "$1" --name aiops >/dev/null 2>&1
  if ! docker exec $NODE crictl inspecti "$1" >/dev/null 2>&1; then
    echo "kind load incomplete for $1, importing with ctr"
    docker save "$1" | docker exec -i $NODE ctr -n k8s.io images import --all-platforms - >/dev/null
  fi
  docker exec $NODE crictl inspecti "$1" >/dev/null 2>&1 && echo "image $1 present on $NODE" || echo "WARNING: $1 missing on $NODE"
}

log "Setup: packages, swap, kind, kubectl"
sudo chmod 666 /var/run/docker.sock
retry 5 sudo apt-get -o DPkg::Lock::Timeout=300 update -qq
retry 3 sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3-venv python3-pip jq curl ca-certificates nano
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
fi
free -h; nproc; df -h / | tail -1
python3 --version; docker --version
ss -ltn | grep -qE ":(5000|8000|8080) " && echo "WARNING: one of ports 5000/8000/8080 is already in use"
retry 5 curl -fsSLo /tmp/kind https://kind.sigs.k8s.io/dl/v0.23.0/kind-linux-amd64
sudo install -m 0755 /tmp/kind /usr/local/bin/kind
retry 5 curl -fsSLo /tmp/kubectl https://dl.k8s.io/release/v1.30.2/bin/linux/amd64/kubectl
sudo install -m 0755 /tmp/kubectl /usr/local/bin/kubectl
kind version; kubectl version --client | head -1

log "Project, venv, pinned dependencies"
mkdir -p ~/aiops-lab/{data,ml,serving/app,serving/tests,k8s,scripts,reports,mlflow}
cd ~/aiops-lab
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==24.0 "setuptools<70" wheel
cat <<'EOF' > requirements-dev.txt
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.4.2
joblib==1.4.2
mlflow==2.12.2
fastapi==0.110.3
uvicorn==0.29.0
pydantic==2.7.1
prometheus-client==0.20.0
pytest==8.2.0
httpx==0.27.0
requests==2.31.0
sqlalchemy==2.0.30
alembic==1.13.1
EOF
retry 3 pip install -r requirements-dev.txt
python -c "import sklearn, mlflow, fastapi; print(sklearn.__version__, mlflow.__version__, fastapi.__version__)"
cat <<'EOF' > ~/aiops-lab/env.sh
cd ~/aiops-lab
source ~/aiops-lab/.venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
EOF
source ~/aiops-lab/env.sh
echo "$MLFLOW_TRACKING_URI"

log "Task 1: versioned, validated datasets"
cat <<'EOF' > ml/common.py
"""Shared constants and helpers for the Incident Predictor project."""
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "cpu_util",        # percent
    "mem_util",        # percent
    "io_wait",         # percent
    "p95_latency_ms",  # milliseconds
    "req_per_sec",     # requests per second
    "error_rate",      # percent of 5xx responses
    "gc_pause_ms",     # milliseconds per minute
]
TARGET = "incident"
MODEL_NAME = "incident-predictor"
EXPERIMENT = "incident-predictor"

# Valid physical ranges, shared by data validation and the API schema.
RANGES = {
    "cpu_util": (0, 100),
    "mem_util": (0, 100),
    "io_wait": (0, 100),
    "p95_latency_ms": (0, 60000),
    "req_per_sec": (0, 100000),
    "error_rate": (0, 100),
    "gc_pause_ms": (0, 10000),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_splits(path, seed=42):
    """Deterministic, stratified 60/20/20 train/val/test split."""
    df = pd.read_csv(path)
    train, temp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[TARGET], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
EOF
cat <<'EOF' > ml/gen_data.py
#!/usr/bin/env python3
"""Generate synthetic service telemetry with incident labels."""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from common import FEATURES, TARGET, sha256_file


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(rows, seed, drifted):
    rng = np.random.default_rng(seed)
    cpu = np.clip(rng.normal(68 if drifted else 55, 15, rows), 0, 100)
    mem = np.clip(rng.normal(60, 12, rows), 0, 100)
    io_wait = np.clip(rng.gamma(2.0, 3.5 if drifted else 2.0, rows), 0, 60)
    latency = rng.lognormal(np.log(260 if drifted else 180), 0.35, rows)
    rps = np.clip(rng.normal(1100 if drifted else 800, 150, rows), 10, None)
    err = np.clip(rng.beta(1.2, 60, rows) * 100, 0, 100)
    gc = rng.gamma(2.0, 15.0, rows)

    z = (-10.0 + 0.045 * cpu + 0.035 * mem + 0.09 * io_wait + 0.006 * latency
         + 0.55 * err + 0.01 * gc + 1.5 * ((cpu > 75) & (io_wait > 8)))
    if drifted:
        # Concept drift: after the migration, request volume drives incidents.
        z = z + 0.01 * (rps - 1100)
    incident = (rng.random(rows) < sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "cpu_util": cpu, "mem_util": mem, "io_wait": io_wait,
        "p95_latency_ms": latency, "req_per_sec": rps,
        "error_rate": err, "gc_pause_ms": gc,
    }).round(2)
    df[TARGET] = incident
    return df[FEATURES + [TARGET]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--drifted", action="store_true")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"telemetry_{args.version}.csv"
    df = generate(args.rows, args.seed, args.drifted)
    df.to_csv(path, index=False)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    digest = sha256_file(path)
    manifest[args.version] = {
        "file": path.name, "rows": len(df), "seed": args.seed,
        "drifted": args.drifted, "incident_rate": round(float(df[TARGET].mean()), 4),
        "sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {path} rows={len(df)} incident_rate={df[TARGET].mean():.3f} "
          f"sha256={digest[:12]}")


if __name__ == "__main__":
    main()
EOF
python ml/gen_data.py --version v1      --rows 20000 --seed 7
python ml/gen_data.py --version v1_live --rows 2000  --seed 99
python ml/gen_data.py --version v2      --rows 8000  --seed 11  --drifted
python ml/gen_data.py --version v2_live --rows 2000  --seed 123 --drifted
cat <<'EOF' > ml/validate_data.py
#!/usr/bin/env python3
"""Schema, null, range and label checks for telemetry files. Exit 1 on failure."""
import sys

import pandas as pd

from common import FEATURES, RANGES, TARGET

MIN_ROWS = 1000


def validate(path):
    errors = []
    df = pd.read_csv(path)
    expected = FEATURES + [TARGET]
    if list(df.columns) != expected:
        errors.append(f"columns {list(df.columns)} != expected {expected}")
        return df, errors
    if len(df) < MIN_ROWS:
        errors.append(f"only {len(df)} rows (< {MIN_ROWS})")
    for col in FEATURES:
        nulls = int(df[col].isna().sum())
        if nulls:
            errors.append(f"{col}: {nulls} null values")
    for col, (lo, hi) in RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            errors.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    if not set(df[TARGET].unique()) <= {0, 1}:
        errors.append(f"{TARGET} must be 0/1, found {sorted(df[TARGET].unique())}")
    else:
        rate = df[TARGET].mean()
        if not 0.01 <= rate <= 0.6:
            errors.append(f"implausible incident rate {rate:.3f}")
    return df, errors


def main(paths):
    failed = False
    for path in paths:
        df, errors = validate(path)
        if errors:
            failed = True
            print(f"[FAIL] {path}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"[PASS] {path} rows={len(df)} incident_rate={df[TARGET].mean():.3f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
EOF
python ml/validate_data.py data/telemetry_v1.csv data/telemetry_v2.csv data/telemetry_v1_live.csv data/telemetry_v2_live.csv
python - <<'EOF'
import numpy as np, pandas as pd
df = pd.read_csv("data/telemetry_v1.csv")
df.loc[0:4, "cpu_util"] = 250
df.loc[5:9, "error_rate"] = np.nan
df.to_csv("/tmp/telemetry_bad.csv", index=False)
EOF
python ml/validate_data.py /tmp/telemetry_bad.csv; echo "exit=$?"
rm -f /tmp/telemetry_bad.csv
jq 'to_entries[] | {version: .key, rows: .value.rows, rate: .value.incident_rate, sha: .value.sha256[0:12]}' data/manifest.json
sha256sum data/telemetry_v1.csv | cut -c1-12

log "Task 2: MLflow server and hyperparameter search"
cat <<'EOF' > scripts/start_mlflow.sh
#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/aiops-lab/mlflow"
mkdir -p "$BASE/artifacts"
if curl -sf http://127.0.0.1:5000/health >/dev/null; then
  echo "MLflow already running"; exit 0
fi
source "$HOME/aiops-lab/.venv/bin/activate"
nohup mlflow server \
  --backend-store-uri "sqlite:///$BASE/mlflow.db" \
  --artifacts-destination "$BASE/artifacts" \
  --host 127.0.0.1 --port 5000 --workers 1 \
  > "$BASE/server.log" 2>&1 &
echo "Waiting for MLflow..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5000/health >/dev/null; then
    echo "MLflow is up on http://127.0.0.1:5000"; exit 0
  fi
  sleep 2
done
echo "MLflow did not start; see $BASE/server.log"; exit 1
EOF
chmod +x scripts/*.sh
./scripts/start_mlflow.sh || { echo "MLflow start failed, log:"; tail -30 ~/aiops-lab/mlflow/server.log; }
curl -s http://127.0.0.1:5000/health; echo
cat <<'EOF' > ml/train.py
#!/usr/bin/env python3
"""Hyperparameter search for the Incident Predictor with MLflow tracking."""
import argparse
import pathlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import EXPERIMENT, FEATURES, TARGET, load_splits, sha256_file

SEARCH_SPACE = [
    ("logreg", LogisticRegression,
     {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}),
    ("logreg", LogisticRegression,
     {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    ("random_forest", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5,
      "class_weight": "balanced", "n_jobs": 2, "random_state": 42}),
    ("random_forest", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5,
      "class_weight": "balanced", "n_jobs": 2, "random_state": 42}),
    ("hist_gb", HistGradientBoostingClassifier,
     {"learning_rate": 0.05, "max_depth": 3, "max_iter": 300, "random_state": 42}),
    ("hist_gb", HistGradientBoostingClassifier,
     {"learning_rate": 0.1, "max_depth": 6, "max_iter": 200, "random_state": 42}),
]
PIP_REQS = ["scikit-learn==1.4.2", "pandas==2.2.2", "numpy==1.26.4"]


def best_threshold(y_true, proba):
    """Threshold that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="append", required=True,
                   help="CSV file; repeat to combine datasets")
    p.add_argument("--data-version", required=True)
    p.add_argument("--experiment", default=EXPERIMENT)
    args = p.parse_args()

    splits = [load_splits(path) for path in args.data]
    train = pd.concat([s[0] for s in splits], ignore_index=True)
    val = pd.concat([s[1] for s in splits], ignore_index=True)
    test = pd.concat([s[2] for s in splits], ignore_index=True)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    hashes = {pathlib.Path(path).name: sha256_file(path)[:12] for path in args.data}

    mlflow.set_experiment(args.experiment)
    results = []
    for idx, (family, cls, params) in enumerate(SEARCH_SPACE, start=1):
        run_name = f"{args.data_version}-{family}-{idx}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"data_version": args.data_version,
                             "model_family": family, "stage": "experiment"})
            mlflow.log_params({"model_family": family,
                               "train_rows": len(train), "val_rows": len(val),
                               **{f"clf__{k}": v for k, v in params.items()},
                               **{f"data_sha256.{k}": v for k, v in hashes.items()}})

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", cls(**params))])
            pipe.fit(X_train, y_train)

            val_proba = pipe.predict_proba(X_val)[:, 1]
            thr, val_f1 = best_threshold(y_val, val_proba)
            val_pred = (val_proba >= thr).astype(int)
            test_proba = pipe.predict_proba(X_test)[:, 1]
            test_pred = (test_proba >= thr).astype(int)

            metrics = {
                "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
                "val_pr_auc": float(average_precision_score(y_val, val_proba)),
                "val_best_threshold": thr,
                "val_f1": val_f1,
                "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
                "test_pr_auc": float(average_precision_score(y_test, test_proba)),
                "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"labels": [0, 1], "threshold": thr,
                             "matrix": confusion_matrix(y_test, test_pred).tolist()},
                            "eval/test_confusion_matrix.json")
            signature = infer_signature(X_val.head(50), pipe.predict(X_val.head(50)))
            mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature,
                                     input_example=X_val.head(3), pip_requirements=PIP_REQS)
            results.append((run_name, run.info.run_id, metrics))
            print(f"{run_name:<28} val_pr_auc={metrics['val_pr_auc']:.4f} "
                  f"val_roc_auc={metrics['val_roc_auc']:.4f} "
                  f"threshold={thr:.3f} val_f1={val_f1:.4f}")

    best = max(results, key=lambda r: r[2]["val_pr_auc"])
    print(f"\nBest run: {best[0]} ({best[1]}) val_pr_auc={best[2]['val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
EOF
python ml/validate_data.py data/telemetry_v1.csv && python ml/train.py --data data/telemetry_v1.csv --data-version v1 2>&1 | grep -v -i warning
EXP_ID=$(curl -s "http://127.0.0.1:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=incident-predictor" | jq -r .experiment.experiment_id)
echo "experiment_id=$EXP_ID"
mlflow runs list --experiment-id "$EXP_ID"
curl -s -X POST http://127.0.0.1:5000/api/2.0/mlflow/runs/search \
  -H 'Content-Type: application/json' \
  -d "{\"experiment_ids\":[\"$EXP_ID\"],\"filter\":\"tags.data_version = 'v1'\",\"order_by\":[\"metrics.val_pr_auc DESC\"],\"max_results\":3}" \
  | jq -r '.runs[] | [.info.run_name, (.data.metrics[] | select(.key=="val_pr_auc") | .value)] | @tsv'
python - <<'EOF'
import mlflow
df = mlflow.search_runs(experiment_names=["incident-predictor"],
                        filter_string="tags.data_version = 'v1'")
cols = ["tags.mlflow.runName", "metrics.val_pr_auc", "metrics.val_roc_auc",
        "metrics.val_best_threshold", "metrics.test_pr_auc"]
print(df[cols].sort_values("metrics.val_pr_auc", ascending=False).to_string(index=False))
EOF
python - <<'EOF'
import mlflow
df = mlflow.search_runs(experiment_names=["incident-predictor"],
                        filter_string="tags.data_version = 'v1' and attributes.status = 'FINISHED'")
print("finished v1 runs:", len(df))
run_id = df.sort_values("metrics.val_pr_auc", ascending=False).iloc[0].run_id
arts = [a.path for a in mlflow.MlflowClient().list_artifacts(run_id, "model")]
print("model artifacts:", sorted(arts))
EOF

log "Task 3: registry, aliases, export"
cat <<'EOF' > ml/register.py
#!/usr/bin/env python3
"""Register the best run for a data version and point an alias at it."""
import argparse
import sys

import mlflow
from mlflow import MlflowClient

from common import EXPERIMENT, MODEL_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-version", required=True)
    p.add_argument("--alias", required=True)
    args = p.parse_args()

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.data_version = '{args.data_version}' and attributes.status = 'FINISHED'",
        order_by=["metrics.val_pr_auc DESC"], max_results=1)
    if runs.empty:
        sys.exit(f"No finished runs with data_version={args.data_version}")
    best = runs.iloc[0]
    run_id, run_name, score = best["run_id"], best["tags.mlflow.runName"], best["metrics.val_pr_auc"]

    client = MlflowClient()
    existing = [mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
                if mv.run_id == run_id]
    if existing:
        mv = existing[0]
        print(f"Run {run_id} already registered as version {mv.version}")
    else:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    client.set_model_version_tag(MODEL_NAME, mv.version, "data_version", args.data_version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "val_pr_auc", f"{score:.4f}")
    client.update_model_version(MODEL_NAME, mv.version,
                                description=f"Best {args.data_version} run {run_name}")
    client.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)
    print(f"{MODEL_NAME} version {mv.version} <- run {run_name} "
          f"(val_pr_auc={score:.4f}); alias '{args.alias}' set")


if __name__ == "__main__":
    main()
EOF
python ml/register.py --data-version v1 --alias champion
python ml/register.py --data-version v1 --alias champion
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/alias?name=incident-predictor&alias=champion" \
  | jq '.model_version | {name, version, run_id, aliases, tags}'
cat <<'EOF' > ml/export.py
#!/usr/bin/env python3
"""Export a registry alias to a self-contained serving directory."""
import argparse
import datetime as dt
import json
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient

from common import FEATURES, MODEL_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default="champion")
    p.add_argument("--out", default="serving/model")
    args = p.parse_args()

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, args.alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{args.alias}")
    run = client.get_run(mv.run_id)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {
        "model_name": MODEL_NAME,
        "model_version": str(mv.version),
        "alias": args.alias,
        "run_id": mv.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "data_version": run.data.tags.get("data_version"),
        "threshold": run.data.metrics["val_best_threshold"],
        "val_pr_auc": run.data.metrics["val_pr_auc"],
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Round-trip check: the exported file must predict exactly like the registry model.
    sample = pd.read_csv("data/telemetry_v1.csv").head(200)[FEATURES]
    reloaded = joblib.load(out / "model.joblib")
    same = np.allclose(model.predict_proba(sample), reloaded.predict_proba(sample))
    print(f"exported {MODEL_NAME} v{mv.version} ({meta['run_name']}) to {out}/ "
          f"threshold={meta['threshold']:.4f} roundtrip_ok={same}")


if __name__ == "__main__":
    main()
EOF
python ml/export.py --alias champion --out serving/model
ls -la serving/model
jq '{model_version, data_version, threshold, sklearn_version}' serving/model/metadata.json
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get?name=incident-predictor" | jq '.registered_model.aliases'
jq -r .model_version serving/model/metadata.json

log "Task 4: FastAPI service and tests"
cat <<'EOF' > serving/requirements.txt
fastapi==0.110.3
uvicorn==0.29.0
pydantic==2.7.1
prometheus-client==0.20.0
scikit-learn==1.4.2
scipy==1.13.1
numpy==1.26.4
pandas==2.2.2
joblib==1.4.2
EOF
touch serving/app/__init__.py
cat <<'EOF' > serving/app/main.py
"""Incident Predictor inference service."""
import json
import logging
import os
import pathlib
import time

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", "/app/model"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("incident-api")

PREDICTIONS = Counter("incident_predictions_total",
                      "Predictions served, by model version and outcome",
                      ["model_version", "outcome"])
PREDICT_LATENCY = Histogram("incident_predict_latency_seconds", "Wall time spent in /predict",
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
PROBABILITY = Histogram("incident_probability", "Predicted incident probability",
                        ["model_version"],
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
BATCH_SIZE = Histogram("incident_batch_size", "Instances per request",
                       buckets=(1, 5, 10, 50, 100, 500, 1000))
MODEL_INFO = Gauge("incident_model_info", "Loaded model metadata (value is always 1)",
                   ["model_name", "model_version", "run_id"])
MODEL_LOADED = Gauge("incident_model_loaded", "1 if a model is loaded and ready")


class _State:
    model = None
    meta: dict = {}
    error = None


STATE = _State()


def load_model():
    try:
        STATE.model = joblib.load(MODEL_DIR / "model.joblib")
        STATE.meta = json.loads((MODEL_DIR / "metadata.json").read_text())
        trained_with = STATE.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning("scikit-learn mismatch: model=%s runtime=%s",
                        trained_with, sklearn.__version__)
        MODEL_INFO.labels(model_name=str(STATE.meta.get("model_name")),
                          model_version=str(STATE.meta.get("model_version")),
                          run_id=str(STATE.meta.get("run_id"))).set(1)
        MODEL_LOADED.set(1)
        log.info("Loaded model %s v%s (threshold=%.4f) from %s",
                 STATE.meta.get("model_name"), STATE.meta.get("model_version"),
                 float(STATE.meta.get("threshold")), MODEL_DIR)
    except Exception as exc:  # keep the process alive; readiness reports the problem
        STATE.model = None
        STATE.error = f"{type(exc).__name__}: {exc}"
        MODEL_LOADED.set(0)
        log.error("Model load failed: %s", STATE.error)


load_model()
app = FastAPI(title="Incident Predictor API", version="1.0.0")


class Telemetry(BaseModel):
    cpu_util: float = Field(..., ge=0, le=100)
    mem_util: float = Field(..., ge=0, le=100)
    io_wait: float = Field(..., ge=0, le=100)
    p95_latency_ms: float = Field(..., gt=0, le=60000)
    req_per_sec: float = Field(..., ge=0, le=100000)
    error_rate: float = Field(..., ge=0, le=100)
    gc_pause_ms: float = Field(..., ge=0, le=10000)


class PredictRequest(BaseModel):
    instances: list[Telemetry] = Field(..., min_length=1, max_length=1000)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    return {"status": "ready", "model_version": str(STATE.meta.get("model_version"))}


@app.get("/model")
def model_info():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    return STATE.meta


@app.post("/predict")
def predict(req: PredictRequest):
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    start = time.perf_counter()
    features = STATE.meta["features"]
    df = pd.DataFrame([item.model_dump() for item in req.instances])[features]
    proba = STATE.model.predict_proba(df)[:, 1]
    threshold = float(STATE.meta["threshold"])
    version = str(STATE.meta["model_version"])

    results = []
    for p in proba:
        flag = bool(p >= threshold)
        PREDICTIONS.labels(model_version=version,
                           outcome="incident" if flag else "normal").inc()
        PROBABILITY.labels(model_version=version).observe(float(p))
        results.append({"incident_probability": round(float(p), 4), "incident": flag})
    BATCH_SIZE.observe(len(results))
    PREDICT_LATENCY.observe(time.perf_counter() - start)
    return {"model_name": STATE.meta["model_name"], "model_version": version,
            "threshold": round(threshold, 4), "predictions": results}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
EOF
cat <<'EOF' > serving/pytest.ini
[pytest]
testpaths = tests
pythonpath = .
EOF
cat <<'EOF' > serving/tests/conftest.py
import os
import pathlib

# Must run before app.main is imported: the model is loaded at import time.
os.environ.setdefault("MODEL_DIR", str(pathlib.Path(__file__).resolve().parents[1] / "model"))
EOF
cat <<'EOF' > serving/tests/test_api.py
from fastapi.testclient import TestClient

from app.main import STATE, app

client = TestClient(app)

HIGH_RISK = {"cpu_util": 92, "mem_util": 85, "io_wait": 14, "p95_latency_ms": 520,
             "req_per_sec": 900, "error_rate": 7, "gc_pause_ms": 70}
LOW_RISK = {"cpu_util": 20, "mem_util": 30, "io_wait": 0.5, "p95_latency_ms": 90,
            "req_per_sec": 500, "error_rate": 0.1, "gc_pause_ms": 5}


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_readyz_reports_loaded_model():
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["model_version"] == str(STATE.meta["model_version"])


def test_model_metadata_contract():
    body = client.get("/model").json()
    assert 0 < body["threshold"] < 1
    assert body["features"] == list(HIGH_RISK.keys())


def test_predict_batch_shape_and_ranges():
    r = client.post("/predict", json={"instances": [HIGH_RISK, LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    assert all(0.0 <= p["incident_probability"] <= 1.0 for p in preds)


def test_behaviour_high_vs_low_risk():
    preds = client.post("/predict", json={"instances": [HIGH_RISK, LOW_RISK]}).json()["predictions"]
    assert preds[0]["incident"] is True
    assert preds[1]["incident"] is False
    assert preds[0]["incident_probability"] > preds[1]["incident_probability"]


def test_rejects_out_of_range_value():
    r = client.post("/predict", json={"instances": [{**HIGH_RISK, "cpu_util": 150}]})
    assert r.status_code == 422


def test_rejects_missing_field():
    bad = {k: v for k, v in HIGH_RISK.items() if k != "error_rate"}
    assert client.post("/predict", json={"instances": [bad]}).status_code == 422


def test_rejects_empty_batch():
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_metrics_exposed():
    client.post("/predict", json={"instances": [LOW_RISK]})
    text = client.get("/metrics").text
    assert "incident_predictions_total" in text
    assert "incident_model_info" in text
EOF
cd ~/aiops-lab/serving && python -m pytest -q 2>&1 | tail -15; cd ~/aiops-lab
cd ~/aiops-lab/serving
MODEL_DIR=$PWD/model nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/uvicorn.log 2>&1 &
UVICORN_PID=$!
cd ~/aiops-lab
wuntil 60 "curl -sf http://127.0.0.1:8000/readyz"
curl -s http://127.0.0.1:8000/readyz | jq -c
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "instances": [
    {"cpu_util":92,"mem_util":85,"io_wait":14,"p95_latency_ms":520,"req_per_sec":900,"error_rate":7,"gc_pause_ms":70},
    {"cpu_util":20,"mem_util":30,"io_wait":0.5,"p95_latency_ms":90,"req_per_sec":500,"error_rate":0.1,"gc_pause_ms":5}
  ]}' | jq
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"instances":[{"cpu_util":150,"mem_util":30,"io_wait":1,"p95_latency_ms":90,"req_per_sec":500,"error_rate":0.1,"gc_pause_ms":5}]}' \
  | jq -c '.detail[0] | {loc, msg}'
curl -s http://127.0.0.1:8000/metrics | grep -E '^incident_(predictions_total|model_info|model_loaded)'
kill $UVICORN_PID; sleep 2
grep -c "Loaded model incident-predictor v1" /tmp/uvicorn.log

log "Task 5: container image"
cat <<'EOF' > serving/.dockerignore
tests/
__pycache__/
*.pyc
.pytest_cache/
pytest.ini
EOF
# base image is 3.12 (not 3.10) so it matches the Python that trained and pickled the model on this VM
cat <<'EOF' > serving/Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app

# Dependencies first so that model-only changes reuse this cached layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY model/ ./model/

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
retry 3 docker build -q --provenance=false --sbom=false -t incident-api:1.0.0 serving/
docker image ls incident-api
docker rm -f incident-api incident-api-broken >/dev/null 2>&1
docker run -d --name incident-api -p 8000:8000 incident-api:1.0.0
sleep 20
docker inspect --format '{{.State.Health.Status}}' incident-api
docker exec incident-api id
curl -s http://127.0.0.1:8000/model | jq -c '{model_version, data_version, sklearn_version}'
docker stats --no-stream --format '{{.Name}} cpu={{.CPUPerc}} mem={{.MemUsage}}' incident-api
docker run -d --name incident-api-broken -p 8001:8000 -e MODEL_DIR=/app/models incident-api:1.0.0
sleep 20
docker inspect --format '{{.State.Health.Status}}' incident-api-broken
curl -s -o /dev/null -w "healthz=%{http_code}\n" http://127.0.0.1:8001/healthz
curl -s -w "\nreadyz=%{http_code}\n" http://127.0.0.1:8001/readyz
docker logs incident-api-broken 2>&1 | grep "Model load failed"
docker rm -f incident-api incident-api-broken >/dev/null
docker image inspect incident-api:1.0.0 --format 'user={{.Config.User}} cmd={{.Config.Cmd}} healthcheck={{if .Config.Healthcheck}}yes{{end}}'
docker run --rm --entrypoint python incident-api:1.0.0 -c "import json; print(json.load(open('/app/model/metadata.json'))['model_version'])"

log "Task 6: kind, probes, safe rollouts"
cat <<'EOF' > k8s/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: aiops
nodes:
  - role: control-plane
EOF
retry 3 bash -c 'kind delete cluster --name aiops >/dev/null 2>&1; kind create cluster --config ~/aiops-lab/k8s/kind-config.yaml --wait 180s'
kubectl config use-context kind-aiops
kubectl get nodes
loadimg incident-api:1.0.0
cat <<'EOF' > k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: aiops-ml
EOF
cat <<'EOF' > k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: incident-api
  namespace: aiops-ml
  labels:
    app: incident-api
spec:
  selector:
    app: incident-api
  ports:
    - name: http
      port: 80
      targetPort: http
EOF
kubectl apply -f k8s/namespace.yaml -f k8s/service.yaml
cat <<'EOF' > k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: incident-api
  namespace: aiops-ml
  labels:
    app: incident-api
spec:
  replicas: 2
  revisionHistoryLimit: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: incident-api
      track: stable
  template:
    metadata:
      labels:
        app: incident-api
        track: stable
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
      containers:
        - name: api
          image: incident-api:1.0.0
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8000
          env:
            - name: LOG_LEVEL
              value: INFO
          resources:
            requests:
              cpu: 100m
              memory: 192Mi
            limits:
              cpu: 500m
              memory: 512Mi
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 2
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
EOF
kubectl apply --dry-run=server -f k8s/deployment.yaml
kubectl apply -f k8s/deployment.yaml
kubectl -n aiops-ml annotate deployment/incident-api kubernetes.io/change-cause="model v1 (data v1), image 1.0.0" --overwrite
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=300s
kubectl -n aiops-ml get pods -l app=incident-api
kubectl -n aiops-ml get endpoints incident-api
cat <<'EOF' > scripts/pf.sh
#!/usr/bin/env bash
# (Re)start a port-forward to the incident-api Service on localhost:8080.
pkill -f "kubectl -n aiops-ml port-forward" 2>/dev/null || true
sleep 1
nohup kubectl -n aiops-ml port-forward svc/incident-api 8080:80 > /tmp/pf.log 2>&1 &
for _ in $(seq 1 15); do
  if curl -sf http://127.0.0.1:8080/readyz >/dev/null; then
    echo "port-forward ready on http://127.0.0.1:8080"; exit 0
  fi
  sleep 1
done
echo "port-forward failed; see /tmp/pf.log"; exit 1
EOF
chmod +x scripts/pf.sh
./scripts/pf.sh
curl -s http://127.0.0.1:8080/readyz | jq -c
echo "--- drill A: image not on the node ---"
kubectl -n aiops-ml set image deployment/incident-api api=incident-api:1.0.1
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=45s; echo "rollout exit=$?"
kubectl -n aiops-ml get pods -l app=incident-api
NEWPOD=$(kubectl -n aiops-ml get pods -l app=incident-api --no-headers | awk '$3!="Running"{print $1; exit}')
kubectl -n aiops-ml describe pod "$NEWPOD" | grep -E "Failed|Back-off" | head -3
curl -s http://127.0.0.1:8080/readyz | jq -c
kubectl -n aiops-ml rollout undo deployment/incident-api
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=180s
echo "--- drill B: bad config caught by readiness ---"
kubectl -n aiops-ml set env deployment/incident-api MODEL_DIR=/app/models
sleep 30
kubectl -n aiops-ml get pods -l app=incident-api
NEWPOD=$(kubectl -n aiops-ml get pods -l app=incident-api --no-headers | awk '$2=="0/1"{print $1; exit}')
kubectl -n aiops-ml describe pod "$NEWPOD" | grep "Readiness probe failed" | head -1
kubectl -n aiops-ml logs "$NEWPOD" | grep "Model load failed"
kubectl -n aiops-ml rollout undo deployment/incident-api
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=180s
kubectl -n aiops-ml rollout history deployment/incident-api
./scripts/pf.sh
kubectl -n aiops-ml get deploy incident-api -o jsonpath='{.spec.template.spec.containers[0].image}{"  ready="}{.status.readyReplicas}{"\n"}'
kubectl -n aiops-ml get deploy incident-api -o jsonpath='{.spec.template.spec.containers[0].readinessProbe.httpGet.path}{" "}{.spec.template.spec.containers[0].livenessProbe.httpGet.path}{"\n"}'
kubectl -n aiops-ml get deploy incident-api -o jsonpath='{.spec.template.spec.containers[0].env}{"\n"}'

log "Task 7: traffic replay, drift detection, impact"
cat <<'EOF' > ml/send_traffic.py
#!/usr/bin/env python3
"""Replay telemetry against the API and log the inference window to CSV."""
import argparse
import json

import numpy as np
import pandas as pd
import requests

from common import FEATURES, TARGET


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--csv", required=True)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--out", required=True, help="where to log the inference window")
    args = p.parse_args()

    df = pd.read_csv(args.csv).head(args.rows)
    flags, versions, errors, requests_sent = [], set(), 0, 0
    for start in range(0, len(df), args.batch):
        chunk = df.iloc[start:start + args.batch]
        payload = {"instances": json.loads(chunk[FEATURES].to_json(orient="records"))}
        r = requests.post(f"{args.url}/predict", json=payload, timeout=15)
        requests_sent += 1
        if r.status_code != 200:
            errors += 1
            continue
        body = r.json()
        versions.add(body["model_version"])
        flags.extend(p["incident"] for p in body["predictions"])

    df[FEATURES].to_csv(args.out, index=False)
    print(f"sent={len(df)} requests={requests_sent} errors={errors} "
          f"model_versions={sorted(versions)}")
    print(f"predicted_incident_rate={np.mean(flags):.3f}")
    if TARGET in df.columns:
        print(f"actual_incident_rate(delayed labels)={df[TARGET].mean():.3f}")
    print(f"inference window logged to {args.out}")


if __name__ == "__main__":
    main()
EOF
./scripts/pf.sh
python ml/send_traffic.py --csv data/telemetry_v1_live.csv --out data/window_baseline.csv
curl -s http://127.0.0.1:8080/metrics | awk '/^incident_predictions_total/{ if ($0 ~ /outcome="incident"/) i+=$2; else n+=$2 } END { printf "cumulative incident_ratio=%.3f (incident=%d normal=%d)\n", i/(i+n), i, n }'
python ml/send_traffic.py --csv data/telemetry_v2_live.csv --out data/window_current.csv
curl -s http://127.0.0.1:8080/metrics | awk '/^incident_predictions_total/{ if ($0 ~ /outcome="incident"/) i+=$2; else n+=$2 } END { printf "cumulative incident_ratio=%.3f (incident=%d normal=%d)\n", i/(i+n), i, n }'
curl -s http://127.0.0.1:8080/metrics | grep -E '^incident_probability_bucket' | head -10
cat <<'EOF' > ml/drift.py
#!/usr/bin/env python3
"""Feature drift report: PSI (primary) and KS test (supporting evidence)."""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from common import FEATURES

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def compute_psi(reference, current, bins=10, eps=1e-6):
    """Population Stability Index of `current` against `reference`."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    n_buckets = len(edges) + 1
    ref_idx = np.searchsorted(edges, reference, side="right")
    cur_idx = np.searchsorted(edges, current, side="right")
    ref_pct = np.bincount(ref_idx, minlength=n_buckets) / len(reference)
    cur_pct = np.bincount(cur_idx, minlength=n_buckets) / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify(psi):
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def self_test():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50000)
    b = rng.normal(1, 1, 50000)
    same = compute_psi(a, a)
    shifted = compute_psi(a, b)
    print(f"psi(identical)={same:.4f}  psi(mean shift of 1 sd)={shifted:.4f}")
    assert abs(same) < 1e-9, "identical samples must give PSI 0"
    assert 0.6 < shifted < 1.1, "a 1-sd mean shift should give PSI roughly 0.8-1.0"
    print("self-test OK")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference")
    p.add_argument("--current")
    p.add_argument("--out")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return

    ref, cur = pd.read_csv(args.reference), pd.read_csv(args.current)
    missing = [f for f in FEATURES if f not in cur.columns]
    if missing:
        sys.exit(f"current window is missing features: {missing}")

    report = {"reference": args.reference, "current": args.current,
              "reference_rows": len(ref), "current_rows": len(cur), "features": {}}
    print(f"{'feature':<16} {'psi':>8} {'ks_pvalue':>10}  status")
    for f in FEATURES:
        psi = compute_psi(ref[f].to_numpy(), cur[f].to_numpy())
        ks_p = float(ks_2samp(ref[f], cur[f]).pvalue)
        status = classify(psi)
        report["features"][f] = {"psi": round(psi, 4), "ks_pvalue": ks_p, "status": status}
        print(f"{f:<16} {psi:>8.4f} {ks_p:>10.4f}  {status}")
    drifted = [f for f, v in report["features"].items() if v["status"] == "significant"]
    report["drift_detected"] = bool(drifted)
    report["significant_features"] = drifted
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\ndrift_detected={bool(drifted)} significant={drifted} -> {args.out}")
    sys.exit(1 if drifted else 0)


if __name__ == "__main__":
    main()
EOF
python ml/drift.py --self-test
python ml/drift.py --reference data/telemetry_v1.csv --current data/window_baseline.csv --out reports/drift_baseline.json; echo "exit=$?"
python ml/drift.py --reference data/telemetry_v1.csv --current data/window_current.csv --out reports/drift_report.json; echo "exit=$?"
cat <<'EOF' > ml/evaluate.py
#!/usr/bin/env python3
"""Evaluate a registry alias on the v1 and v2 held-out test splits."""
import argparse

import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import FEATURES, MODEL_NAME, TARGET, load_splits

DATASETS = [("v1", "data/telemetry_v1.csv"), ("v2", "data/telemetry_v2.csv")]


def evaluate_alias(alias):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    threshold = client.get_run(mv.run_id).data.metrics["val_best_threshold"]
    result = {"alias": alias, "version": str(mv.version), "threshold": threshold}
    for tag, path in DATASETS:
        _, _, test = load_splits(path)
        proba = model.predict_proba(test[FEATURES])[:, 1]
        pred = (proba >= threshold).astype(int)
        result[f"{tag}_roc_auc"] = float(roc_auc_score(test[TARGET], proba))
        result[f"{tag}_pr_auc"] = float(average_precision_score(test[TARGET], proba))
        result[f"{tag}_f1"] = float(f1_score(test[TARGET], pred, zero_division=0))
        result[f"{tag}_pred_rate"] = float(pred.mean())
        result[f"{tag}_true_rate"] = float(test[TARGET].mean())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default="champion")
    args = p.parse_args()
    r = evaluate_alias(args.alias)
    print(f"alias={r['alias']} version={r['version']} threshold={r['threshold']:.4f}")
    print(f"{'metric':<10} {'v1 test':>8} {'v2 test':>8}")
    for m in ["roc_auc", "pr_auc", "f1", "pred_rate", "true_rate"]:
        print(f"{m:<10} {r['v1_' + m]:>8.4f} {r['v2_' + m]:>8.4f}")


if __name__ == "__main__":
    main()
EOF
python ml/evaluate.py --alias champion
jq -r '.significant_features | join(",")' reports/drift_report.json
jq -r '.drift_detected' reports/drift_baseline.json

log "Task 8: retrain, gate, promote, roll out, rollback"
python ml/validate_data.py data/telemetry_v1.csv data/telemetry_v2.csv && \
python ml/train.py --data data/telemetry_v1.csv --data data/telemetry_v2.csv --data-version v1+v2 2>&1 | grep -v -i warning
python ml/register.py --data-version v1+v2 --alias challenger
cat <<'EOF' > ml/promote.py
#!/usr/bin/env python3
"""Champion/challenger quality gate. Exit 0 = promoted, 2 = rejected."""
import argparse
import datetime as dt
import json
import sys

from mlflow import MlflowClient

from common import MODEL_NAME
from evaluate import evaluate_alias


def gate(champion, challenger, min_gain, max_regression):
    """Decide whether the challenger may replace the champion.

    Returns (passed: bool, reasons: list[str]).
    """
    reasons = []

    gain = challenger["v2_pr_auc"] - champion["v2_pr_auc"]
    ok_gain = gain >= min_gain
    reasons.append(f"{'PASS' if ok_gain else 'FAIL'} recent-data gain {gain:+.4f} >= {min_gain:.4f}")

    regression = champion["v1_pr_auc"] - challenger["v1_pr_auc"]
    ok_regression = challenger["v1_pr_auc"] >= champion["v1_pr_auc"] - max_regression
    reasons.append(f"{'PASS' if ok_regression else 'FAIL'} v1 regression {regression:+.4f} <= {max_regression:.4f}")

    return (ok_gain and ok_regression), reasons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--max-regression", type=float, default=0.05)
    args = p.parse_args()

    client = MlflowClient()
    champion = evaluate_alias("champion")
    challenger = evaluate_alias("challenger")
    print(f"{'metric':<10} {'champion v' + champion['version']:>14} "
          f"{'challenger v' + challenger['version']:>16}")
    for m in ["v1_pr_auc", "v2_pr_auc", "v1_roc_auc", "v2_roc_auc", "v2_f1"]:
        print(f"{m:<10} {champion[m]:>14.4f} {challenger[m]:>16.4f}")

    passed, reasons = gate(champion, challenger, args.min_gain, args.max_regression)
    for r in reasons:
        print(" ", r)

    report = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "champion": champion, "challenger": challenger, "passed": passed,
              "reasons": reasons, "min_gain": args.min_gain,
              "max_regression": args.max_regression}
    with open("reports/promotion_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    if not passed:
        client.set_model_version_tag(MODEL_NAME, challenger["version"], "promotion", "rejected")
        print(f"REJECTED: champion stays at version {champion['version']}")
        sys.exit(2)

    client.set_registered_model_alias(MODEL_NAME, "previous", champion["version"])
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger["version"])
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    client.set_model_version_tag(MODEL_NAME, challenger["version"], "promotion", "approved")
    print(f"PROMOTED: champion -> version {challenger['version']} "
          f"(previous -> version {champion['version']})")


if __name__ == "__main__":
    main()
EOF
python ml/promote.py; PRC=$?; echo "exit=$PRC"
if [ $PRC -ne 0 ]; then
  echo "NOTE: default gate rejected the challenger; applying the documented override --max-regression 0.08 (record this in the change record)"
  python ml/promote.py --max-regression 0.08; echo "exit=$?"
fi
curl -s "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get?name=incident-predictor" | jq -c '.registered_model.aliases'
python ml/export.py --alias champion --out serving/model
(cd serving && python -m pytest -q 2>&1 | tail -5) && retry 3 docker build -q --provenance=false --sbom=false -t incident-api:1.1.0 serving/
loadimg incident-api:1.1.0
kubectl -n aiops-ml set image deployment/incident-api api=incident-api:1.1.0
kubectl -n aiops-ml annotate deployment/incident-api kubernetes.io/change-cause="model v2 (data v1+v2), image 1.1.0" --overwrite
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=300s
./scripts/pf.sh
curl -s http://127.0.0.1:8080/model | jq -c '{model_version, data_version, run_name}'
for pod in $(kubectl -n aiops-ml get pods -l app=incident-api -o name); do
  echo -n "$pod -> "
  kubectl -n aiops-ml exec "$pod" -- python -c "import json,urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/model'))['model_version'])"
done
echo "--- rollback drill ---"
kubectl -n aiops-ml rollout undo deployment/incident-api
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=300s
./scripts/pf.sh
curl -s http://127.0.0.1:8080/readyz | jq -c
python -c "from mlflow import MlflowClient as C; print('registry previous =', C().get_model_version_by_alias('incident-predictor','previous').version)"
kubectl -n aiops-ml set image deployment/incident-api api=incident-api:1.1.0
kubectl -n aiops-ml annotate deployment/incident-api kubernetes.io/change-cause="roll forward: model v2, image 1.1.0" --overwrite
kubectl -n aiops-ml rollout status deployment/incident-api --timeout=300s
./scripts/pf.sh
python ml/send_traffic.py --csv data/telemetry_v2_live.csv --out /tmp/window_v2_after.csv
kubectl -n aiops-ml get deploy incident-api -o jsonpath='{.spec.template.spec.containers[0].image}{"  ready="}{.status.readyReplicas}{"\n"}'
jq -r '.passed, (.reasons|join(" | "))' reports/promotion_report.json
curl -s http://127.0.0.1:8080/metrics | grep '^incident_model_info'

log "Validation (lab validator)"
cat <<'EOF' > scripts/validate.sh
#!/usr/bin/env bash
set -u
cd ~/aiops-lab
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PASS=0; FAIL=0
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "[PASS] $desc"; PASS=$((PASS+1));
  else echo "[FAIL] $desc"; FAIL=$((FAIL+1)); fi
}

check "T1 four datasets generated" test -s data/telemetry_v1.csv -a -s data/telemetry_v2.csv -a -s data/telemetry_v1_live.csv -a -s data/telemetry_v2_live.csv
check "T1 manifest records >= 4 versions with hashes" bash -c '[ "$(jq "[.[] | select(.sha256 | length == 64)] | length" data/manifest.json)" -ge 4 ]'
check "T1 data validation passes on training data" python ml/validate_data.py data/telemetry_v1.csv data/telemetry_v2.csv
check "T2 MLflow tracking server healthy" curl -sf http://127.0.0.1:5000/health
check "T2 >= 12 finished experiment runs" python -c "import mlflow,sys; df=mlflow.search_runs(experiment_names=['incident-predictor'], filter_string=\"attributes.status = 'FINISHED'\"); sys.exit(0 if len(df)>=12 else 1)"
check "T3/T8 registry: champion=2, previous=1" python -c "import sys; from mlflow import MlflowClient as C; c=C(); sys.exit(0 if str(c.get_model_version_by_alias('incident-predictor','champion').version)=='2' and str(c.get_model_version_by_alias('incident-predictor','previous').version)=='1' else 1)"
check "T3/T8 exported artifact is model v2" bash -c '[ "$(jq -r .model_version serving/model/metadata.json)" = "2" ] && test -s serving/model/model.joblib'
check "T4 API unit tests pass" bash -c 'cd serving && python -m pytest -q'
check "T5 images 1.0.0 and 1.1.0 exist" docker image inspect incident-api:1.0.0 incident-api:1.1.0
check "T5 image runs as non-root" bash -c '[ "$(docker image inspect incident-api:1.1.0 --format "{{.Config.User}}")" = "10001" ]'
check "T6 kind cluster aiops exists" bash -c 'kind get clusters | grep -qx aiops'
check "T6 readiness probe on /readyz" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].readinessProbe.httpGet.path}")" = "/readyz" ]'
check "T6 liveness probe on /healthz" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].livenessProbe.httpGet.path}")" = "/healthz" ]'
check "T6 memory limit set" bash -c '[ -n "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].resources.limits.memory}")" ]'
check "T7 PSI self-test passes" python ml/drift.py --self-test
check "T7 baseline window shows no significant drift" jq -e '.drift_detected == false' reports/drift_baseline.json
check "T7 current window flags req_per_sec drift" jq -e '.features.req_per_sec.status == "significant"' reports/drift_report.json
check "T8 promotion gate approved" jq -e '.passed == true' reports/promotion_report.json
check "T8 deployment runs incident-api:1.1.0 with 2 ready replicas" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].image}/{.status.readyReplicas}")" = "incident-api:1.1.0/2" ]'
check "T8 pods serve model version 2" bash -c 'kubectl -n aiops-ml exec deploy/incident-api -- python -c "import urllib.request,json;print(json.load(urllib.request.urlopen(\"http://127.0.0.1:8000/model\"))[\"model_version\"])" | grep -qx 2'

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
EOF
chmod +x scripts/validate.sh
./scripts/validate.sh
echo "LAB45 DONE"
# END_OF_LAB45
