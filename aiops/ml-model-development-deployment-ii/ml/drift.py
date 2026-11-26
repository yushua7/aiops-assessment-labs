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
