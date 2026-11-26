#!/usr/bin/env python3
"""Export a registry alias to a self-contained serving directory."""
import argparse
import datetime as dt
import json
import pathlib

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow import MlflowClient

from common import FEATURES, MODEL_NAME


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alias", default="champion")
    p.add_argument("--out", default="serving/model")
    args = p.parse_args()

    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, args.alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{args.alias}")
    run = client.get_run(mv.run_id)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {
        "model_name": MODEL_NAME,
        "model_version": str(mv.version),
        "alias": args.alias,
        "run_id": mv.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "data_version": run.data.tags.get("data_version"),
        "threshold": run.data.metrics["val_best_threshold"],
        "val_pr_auc": run.data.metrics["val_pr_auc"],
        "features": FEATURES,
        "sklearn_version": sklearn.__version__,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Round-trip check: the exported file must predict exactly like the registry model.
    sample = pd.read_csv("data/telemetry_v1.csv").head(200)[FEATURES]
    reloaded = joblib.load(out / "model.joblib")
    same = np.allclose(model.predict_proba(sample), reloaded.predict_proba(sample))
    print(f"exported {MODEL_NAME} v{mv.version} ({meta['run_name']}) to {out}/ "
          f"threshold={meta['threshold']:.4f} roundtrip_ok={same}")


if __name__ == "__main__":
    main()
