"""Shared constants and helpers for the Incident Predictor project."""
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "cpu_util",        # percent
    "mem_util",        # percent
    "io_wait",         # percent
    "p95_latency_ms",  # milliseconds
    "req_per_sec",     # requests per second
    "error_rate",      # percent of 5xx responses
    "gc_pause_ms",     # milliseconds per minute
]
TARGET = "incident"
MODEL_NAME = "incident-predictor"
EXPERIMENT = "incident-predictor"

# Valid physical ranges, shared by data validation and the API schema.
RANGES = {
    "cpu_util": (0, 100),
    "mem_util": (0, 100),
    "io_wait": (0, 100),
    "p95_latency_ms": (0, 60000),
    "req_per_sec": (0, 100000),
    "error_rate": (0, 100),
    "gc_pause_ms": (0, 10000),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_splits(path, seed=42):
    """Deterministic, stratified 60/20/20 train/val/test split."""
    df = pd.read_csv(path)
    train, temp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[TARGET], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))
