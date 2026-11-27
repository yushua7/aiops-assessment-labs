#!/usr/bin/env python3
"""Evaluate champion vs challenger on the same held-out data; promote if clearly better."""
import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from eval_utils import evaluate
from features import build_features, load_metrics, time_split

MODEL_NAME = "infra-anomaly-detector"


def score_alias(client, alias, Xte, yte):
    mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    thr = float(mv.tags["threshold"])
    pred = (-model.score_samples(Xte) >= thr).astype(int)
    m = evaluate(yte, pred)
    m.update(alias=alias, version=mv.version, candidate=mv.tags.get("candidate"),
             threshold=round(thr, 4), alert_rate=round(float(pred.mean()), 4))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics_v2.csv")
    ap.add_argument("--min-gain", type=float, default=0.02, help="required F1 improvement")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    df = load_metrics(args.data)
    X = build_features(df)
    _, _, te = time_split(df)
    Xte, yte = X.loc[te.index], te["is_anomaly"].to_numpy()
    champ = score_alias(client, "champion", Xte, yte)
    chal = score_alias(client, "challenger", Xte, yte)
    for m in (champ, chal):
        print(f"{m['alias']:<10} v{m['version']:<3} {str(m['candidate']):<20} f1={m['f1']:.3f} "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"false_alerts/day={m['false_alerts_per_day']:.1f} alert_rate={m['alert_rate']:.1%}")

    if chal["f1"] >= champ["f1"] + args.min_gain:
        client.set_registered_model_alias(MODEL_NAME, "previous", champ["version"])
        client.set_registered_model_alias(MODEL_NAME, "champion", chal["version"])
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        decision = "promoted"
        print(f"\nDECISION: promote v{chal['version']} to champion (v{champ['version']} -> 'previous')")
    else:
        decision = "kept"
        print(f"\nDECISION: keep v{champ['version']}; challenger gain below {args.min_gain}")
    Path("results/promotion.json").write_text(json.dumps(
        {"decision": decision, "data": args.data, "champion_before": champ, "challenger": chal}, indent=2))


if __name__ == "__main__":
    main()
