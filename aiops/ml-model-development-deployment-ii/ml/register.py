#!/usr/bin/env python3
"""Register the best run for a data version and point an alias at it."""
import argparse
import sys

import mlflow
from mlflow import MlflowClient

from common import EXPERIMENT, MODEL_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-version", required=True)
    p.add_argument("--alias", required=True)
    args = p.parse_args()

    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        filter_string=f"tags.data_version = '{args.data_version}' and attributes.status = 'FINISHED'",
        order_by=["metrics.val_pr_auc DESC"], max_results=1)
    if runs.empty:
        sys.exit(f"No finished runs with data_version={args.data_version}")
    best = runs.iloc[0]
    run_id, run_name, score = best["run_id"], best["tags.mlflow.runName"], best["metrics.val_pr_auc"]

    client = MlflowClient()
    existing = [mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
                if mv.run_id == run_id]
    if existing:
        mv = existing[0]
        print(f"Run {run_id} already registered as version {mv.version}")
    else:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    client.set_model_version_tag(MODEL_NAME, mv.version, "data_version", args.data_version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "val_pr_auc", f"{score:.4f}")
    client.update_model_version(MODEL_NAME, mv.version,
                                description=f"Best {args.data_version} run {run_name}")
    client.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)
    print(f"{MODEL_NAME} version {mv.version} <- run {run_name} "
          f"(val_pr_auc={score:.4f}); alias '{args.alias}' set")


if __name__ == "__main__":
    main()
