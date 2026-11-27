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
