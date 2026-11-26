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
