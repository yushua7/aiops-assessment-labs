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
