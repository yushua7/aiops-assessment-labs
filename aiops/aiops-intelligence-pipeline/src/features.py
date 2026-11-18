"""Feature engineering shared by training and serving."""
import numpy as np
import pandas as pd

METRICS = ["cpu", "mem", "latency_ms", "error_rate", "req_rate"]


def build_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["timestamp"])
    for m in METRICS:
        x = df[m].astype(float)
        out[m] = x
        prev = x.shift(1).rolling(window, min_periods=1)
        mean = prev.mean().fillna(x)
        std = prev.std().fillna(0.0)
        out[f"{m}_roll_mean"] = mean
        out[f"{m}_roll_std"] = std
        out[f"{m}_zscore"] = ((x - mean) / (std + 1e-6)).clip(-50, 50)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
