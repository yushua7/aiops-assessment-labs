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
