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
