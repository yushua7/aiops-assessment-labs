"""Evaluation helpers for operational anomaly detection."""
import numpy as np
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support


def _runs(mask):
    """Return (start, end_exclusive) tuples of contiguous True runs."""
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def evaluate(y_true, y_pred, steps_per_day=1440):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    incidents = _runs(y_true)
    detected = sum(1 for s, e in incidents if y_pred[s:e].any())
    false_alerts = sum(1 for s, e in _runs(y_pred) if not y_true[s:e].any())
    days = len(y_true) / steps_per_day
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "event_recall": round(detected / len(incidents), 4) if incidents else 0.0,
        "incidents": len(incidents),
        "false_alerts_per_day": round(false_alerts / days, 2),
    }


def best_threshold(scores, y_true):
    """Threshold that maximises F1 (prediction rule: score >= threshold)."""
    p, r, t = precision_recall_curve(np.asarray(y_true).astype(int), scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])
