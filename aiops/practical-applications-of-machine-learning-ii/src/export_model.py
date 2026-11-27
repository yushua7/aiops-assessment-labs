#!/usr/bin/env python3
"""Export a registry alias to a self-contained serving bundle (model.joblib + metadata.json)."""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from features import FEATURE_COLS, WINDOW

MODEL_NAME = "infra-anomaly-detector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", default="service/model")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, args.alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    meta = {"model_name": MODEL_NAME, "version": mv.version, "alias": args.alias,
            "threshold": float(mv.tags["threshold"]), "candidate": mv.tags.get("candidate"),
            "run_id": mv.run_id, "feature_cols": FEATURE_COLS, "window": WINDOW}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported {MODEL_NAME} v{mv.version} ({meta['candidate']}, alias={args.alias}) "
          f"threshold={meta['threshold']:.4f} -> {out}")


if __name__ == "__main__":
    main()
