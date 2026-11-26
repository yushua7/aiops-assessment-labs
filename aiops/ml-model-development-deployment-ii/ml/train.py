#!/usr/bin/env python3
"""Hyperparameter search for the Incident Predictor with MLflow tracking."""
import argparse
import pathlib

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import EXPERIMENT, FEATURES, TARGET, load_splits, sha256_file

SEARCH_SPACE = [
    ("logreg", LogisticRegression,
     {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}),
    ("logreg", LogisticRegression,
     {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    ("random_forest", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5,
      "class_weight": "balanced", "n_jobs": 2, "random_state": 42}),
    ("random_forest", RandomForestClassifier,
     {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 5,
      "class_weight": "balanced", "n_jobs": 2, "random_state": 42}),
    ("hist_gb", HistGradientBoostingClassifier,
     {"learning_rate": 0.05, "max_depth": 3, "max_iter": 300, "random_state": 42}),
    ("hist_gb", HistGradientBoostingClassifier,
     {"learning_rate": 0.1, "max_depth": 6, "max_iter": 200, "random_state": 42}),
]
PIP_REQS = ["scikit-learn==1.4.2", "pandas==2.2.2", "numpy==1.26.4"]


def best_threshold(y_true, proba):
    """Threshold that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="append", required=True,
                   help="CSV file; repeat to combine datasets")
    p.add_argument("--data-version", required=True)
    p.add_argument("--experiment", default=EXPERIMENT)
    args = p.parse_args()

    splits = [load_splits(path) for path in args.data]
    train = pd.concat([s[0] for s in splits], ignore_index=True)
    val = pd.concat([s[1] for s in splits], ignore_index=True)
    test = pd.concat([s[2] for s in splits], ignore_index=True)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    hashes = {pathlib.Path(path).name: sha256_file(path)[:12] for path in args.data}

    mlflow.set_experiment(args.experiment)
    results = []
    for idx, (family, cls, params) in enumerate(SEARCH_SPACE, start=1):
        run_name = f"{args.data_version}-{family}-{idx}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags({"data_version": args.data_version,
                             "model_family": family, "stage": "experiment"})
            mlflow.log_params({"model_family": family,
                               "train_rows": len(train), "val_rows": len(val),
                               **{f"clf__{k}": v for k, v in params.items()},
                               **{f"data_sha256.{k}": v for k, v in hashes.items()}})

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", cls(**params))])
            pipe.fit(X_train, y_train)

            val_proba = pipe.predict_proba(X_val)[:, 1]
            thr, val_f1 = best_threshold(y_val, val_proba)
            val_pred = (val_proba >= thr).astype(int)
            test_proba = pipe.predict_proba(X_test)[:, 1]
            test_pred = (test_proba >= thr).astype(int)

            metrics = {
                "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
                "val_pr_auc": float(average_precision_score(y_val, val_proba)),
                "val_best_threshold": thr,
                "val_f1": val_f1,
                "val_precision": float(precision_score(y_val, val_pred, zero_division=0)),
                "val_recall": float(recall_score(y_val, val_pred, zero_division=0)),
                "test_pr_auc": float(average_precision_score(y_test, test_proba)),
                "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"labels": [0, 1], "threshold": thr,
                             "matrix": confusion_matrix(y_test, test_pred).tolist()},
                            "eval/test_confusion_matrix.json")
            signature = infer_signature(X_val.head(50), pipe.predict(X_val.head(50)))
            mlflow.sklearn.log_model(pipe, artifact_path="model", signature=signature,
                                     input_example=X_val.head(3), pip_requirements=PIP_REQS)
            results.append((run_name, run.info.run_id, metrics))
            print(f"{run_name:<28} val_pr_auc={metrics['val_pr_auc']:.4f} "
                  f"val_roc_auc={metrics['val_roc_auc']:.4f} "
                  f"threshold={thr:.3f} val_f1={val_f1:.4f}")

    best = max(results, key=lambda r: r[2]["val_pr_auc"])
    print(f"\nBest run: {best[0]} ({best[1]}) val_pr_auc={best[2]['val_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
