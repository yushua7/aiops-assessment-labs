#!/usr/bin/env python3
"""Train and compare unsupervised anomaly detectors; track in MLflow; register best."""
import argparse

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval_utils import best_threshold, evaluate
from features import build_features, load_metrics, time_split

MODEL_NAME = "infra-anomaly-detector"
EXPERIMENT = "infra-anomaly"
PIP_REQS = ["scikit-learn==1.4.2", "numpy==1.26.4", "pandas==2.2.2"]


def candidates():
    cands = []
    for n_est in (100, 200):
        for max_samples in (256, 1024):
            cands.append((
                f"iforest_n{n_est}_s{max_samples}",
                {"model": "iforest", "n_estimators": n_est, "max_samples": max_samples},
                IsolationForest(n_estimators=n_est, max_samples=max_samples,
                                random_state=42, n_jobs=-1),
            ))
    for k in (20, 50):
        cands.append((f"lof_k{k}", {"model": "lof", "n_neighbors": k},
                      LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1)))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    ap.add_argument("--run-tag", default="initial")
    ap.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = load_metrics(args.data)
    X = build_features(df)
    tr, va, te = time_split(df)
    Xtr, Xva, Xte = X.loc[tr.index], X.loc[va.index], X.loc[te.index]
    yva, yte = va["is_anomaly"].to_numpy(), te["is_anomaly"].to_numpy()

    results = []
    for name, params, est in candidates():
        with mlflow.start_run(run_name=f"{args.run_tag}-{name}") as run:
            pipe = make_pipeline(StandardScaler(), est)
            pipe.fit(Xtr)
            s_va = -pipe.score_samples(Xva)
            thr, _ = best_threshold(s_va, yva)
            s_te = -pipe.score_samples(Xte)
            m_va = evaluate(yva, (s_va >= thr).astype(int))
            m_te = evaluate(yte, (s_te >= thr).astype(int))

            mlflow.log_params({**params, "threshold": round(thr, 6), "train_rows": len(Xtr)})
            mlflow.set_tags({"run_tag": args.run_tag, "data_file": args.data})
            mlflow.log_metrics({f"val_{k}": v for k, v in m_va.items()}
                               | {f"test_{k}": v for k, v in m_te.items()})
            mlflow.sklearn.log_model(pipe, artifact_path="model",
                                     signature=infer_signature(Xva.head(50), s_va[:50]),
                                     pip_requirements=PIP_REQS)
            results.append({"name": name, "run_id": run.info.run_id, "threshold": thr,
                            "val_f1": m_va["f1"], "test_f1": m_te["f1"]})
            print(f"{name:<22} val_f1={m_va['f1']:.3f} test_f1={m_te['f1']:.3f} "
                  f"test_event_recall={m_te['event_recall']:.2f} "
                  f"test_false_alerts/day={m_te['false_alerts_per_day']:.1f}")

    best = max(results, key=lambda r: r["val_f1"])
    print(f"\nBest candidate by validation F1: {best['name']} (val_f1={best['val_f1']:.3f})")

    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client.set_model_version_tag(MODEL_NAME, mv.version, "threshold", str(best["threshold"]))
    client.set_model_version_tag(MODEL_NAME, mv.version, "candidate", best["name"])
    client.set_model_version_tag(MODEL_NAME, mv.version, "data_file", args.data)
    try:
        client.get_model_version_by_alias(MODEL_NAME, "champion")
        alias = "challenger"
    except MlflowException:
        alias = "champion"
    client.set_registered_model_alias(MODEL_NAME, alias, mv.version)
    print(f"Registered version {mv.version} with alias '{alias}' (threshold={best['threshold']:.4f})")


if __name__ == "__main__":
    main()
