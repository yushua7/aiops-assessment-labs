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
