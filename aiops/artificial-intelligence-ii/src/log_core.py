import numpy as np
import pandas as pd


def rare_templates(train_counts: pd.DataFrame, max_presence: float = 0.02) -> list:
    presence = (train_counts > 0).mean(axis=0)
    return sorted(int(c) for c in presence[presence < max_presence].index)


def fuse(metric_flag, log_flag, mode: str = "or"):
    a = np.asarray(metric_flag, dtype=bool)
    b = np.asarray(log_flag, dtype=bool)
    if mode == "or":
        return a | b
    if mode == "and":
        return a & b
    raise ValueError(f"unknown fusion mode: {mode}")
