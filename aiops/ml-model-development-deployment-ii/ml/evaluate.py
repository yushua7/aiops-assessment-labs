#!/usr/bin/env python3
"""Evaluate a registry alias on the v1 and v2 held-out test splits."""
import argparse

import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from common import FEATURES, MODEL_NAME, TARGET, load_splits

DATASETS = [("v1", "data/telemetry_v1.csv"), ("v2", "data/telemetry_v2.csv")]


def evaluate_alias(alias):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    threshold = client.get_run(mv.run_id).data.metrics["val_best_threshold"]
    result = {"alias": alias, "version": str(mv.version), "threshold": threshold}
    for tag, path in DATASETS:
        _, _, test = load_splits(path)
        proba = model.predict_proba(test[FEATURES])[:, 1]
        pred = (proba >= threshold).astype(int)
        result[f"{tag}_roc_auc"] = float(roc_auc_score(test[TARGET], proba))
        result[f"{tag}_pr_auc"] = float(average_precision_score(test[TARGET], proba))
        result[f"{tag}_f1"] = float(f1_score(test[TARGET], pred, zero_division=0))
        result[f"{tag}_pred_rate"] = float(pred.mean())
        result[f"{tag}_true_rate"] = float(test[TARGET].mean())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default="champion")
    args = p.parse_args()
    r = evaluate_alias(args.alias)
    print(f"alias={r['alias']} version={r['version']} threshold={r['threshold']:.4f}")
    print(f"{'metric':<10} {'v1 test':>8} {'v2 test':>8}")
    for m in ["roc_auc", "pr_auc", "f1", "pred_rate", "true_rate"]:
        print(f"{m:<10} {r['v1_' + m]:>8.4f} {r['v2_' + m]:>8.4f}")


if __name__ == "__main__":
    main()
