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
