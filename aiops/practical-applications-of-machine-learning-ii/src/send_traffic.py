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
