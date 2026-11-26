"""Incident Predictor inference service."""
import json
import logging
import os
import pathlib
import time

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_DIR = pathlib.Path(os.getenv("MODEL_DIR", "/app/model"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("incident-api")

PREDICTIONS = Counter("incident_predictions_total",
                      "Predictions served, by model version and outcome",
                      ["model_version", "outcome"])
PREDICT_LATENCY = Histogram("incident_predict_latency_seconds", "Wall time spent in /predict",
                            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5))
PROBABILITY = Histogram("incident_probability", "Predicted incident probability",
                        ["model_version"],
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
BATCH_SIZE = Histogram("incident_batch_size", "Instances per request",
                       buckets=(1, 5, 10, 50, 100, 500, 1000))
MODEL_INFO = Gauge("incident_model_info", "Loaded model metadata (value is always 1)",
                   ["model_name", "model_version", "run_id"])
MODEL_LOADED = Gauge("incident_model_loaded", "1 if a model is loaded and ready")


class _State:
    model = None
    meta: dict = {}
    error = None


STATE = _State()


def load_model():
    try:
        STATE.model = joblib.load(MODEL_DIR / "model.joblib")
        STATE.meta = json.loads((MODEL_DIR / "metadata.json").read_text())
        trained_with = STATE.meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning("scikit-learn mismatch: model=%s runtime=%s",
                        trained_with, sklearn.__version__)
        MODEL_INFO.labels(model_name=str(STATE.meta.get("model_name")),
                          model_version=str(STATE.meta.get("model_version")),
                          run_id=str(STATE.meta.get("run_id"))).set(1)
        MODEL_LOADED.set(1)
        log.info("Loaded model %s v%s (threshold=%.4f) from %s",
                 STATE.meta.get("model_name"), STATE.meta.get("model_version"),
                 float(STATE.meta.get("threshold")), MODEL_DIR)
    except Exception as exc:  # keep the process alive; readiness reports the problem
        STATE.model = None
        STATE.error = f"{type(exc).__name__}: {exc}"
        MODEL_LOADED.set(0)
        log.error("Model load failed: %s", STATE.error)


load_model()
app = FastAPI(title="Incident Predictor API", version="1.0.0")


class Telemetry(BaseModel):
    cpu_util: float = Field(..., ge=0, le=100)
    mem_util: float = Field(..., ge=0, le=100)
    io_wait: float = Field(..., ge=0, le=100)
    p95_latency_ms: float = Field(..., gt=0, le=60000)
    req_per_sec: float = Field(..., ge=0, le=100000)
    error_rate: float = Field(..., ge=0, le=100)
    gc_pause_ms: float = Field(..., ge=0, le=10000)


class PredictRequest(BaseModel):
    instances: list[Telemetry] = Field(..., min_length=1, max_length=1000)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    return {"status": "ready", "model_version": str(STATE.meta.get("model_version"))}


@app.get("/model")
def model_info():
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    return STATE.meta


@app.post("/predict")
def predict(req: PredictRequest):
    if STATE.model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE.error}")
    start = time.perf_counter()
    features = STATE.meta["features"]
    df = pd.DataFrame([item.model_dump() for item in req.instances])[features]
    proba = STATE.model.predict_proba(df)[:, 1]
    threshold = float(STATE.meta["threshold"])
    version = str(STATE.meta["model_version"])

    results = []
    for p in proba:
        flag = bool(p >= threshold)
        PREDICTIONS.labels(model_version=version,
                           outcome="incident" if flag else "normal").inc()
        PROBABILITY.labels(model_version=version).observe(float(p))
        results.append({"incident_probability": round(float(p), 4), "incident": flag})
    BATCH_SIZE.observe(len(results))
    PREDICT_LATENCY.observe(time.perf_counter() - start)
    return {"model_name": STATE.meta["model_name"], "model_version": version,
            "threshold": round(threshold, 4), "predictions": results}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
