import numpy as np


def psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    k = len(edges) + 1
    ref = np.bincount(np.searchsorted(edges, reference, side="right"), minlength=k) / len(reference)
    cur = np.bincount(np.searchsorted(edges, current, side="right"), minlength=k) / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))
