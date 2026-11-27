"""AIOps anomaly scoring service."""
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from features import BASE_COLS, FEATURE_COLS, build_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))

PREDICTIONS = Counter("anomaly_predictions", "Scored observations by result", ["result"])
for _r in ("anomaly", "normal"):
    PREDICTIONS.labels(result=_r)
INFER_SECONDS = Histogram("anomaly_inference_seconds", "Feature building + scoring time",
                          buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
SCORE = Histogram("anomaly_score", "Distribution of anomaly scores",
                  buckets=(0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0, 5.0))
INPUT_VALUE = Gauge("anomaly_input_value", "Latest raw input value per metric", ["metric"])
MODEL_INFO = Gauge("anomaly_model_info", "Currently loaded model", ["version", "candidate"])
MODEL_LOADED = Gauge("anomaly_model_loaded", "1 if a model is loaded, else 0")

STATE = {"model": None, "meta": {}, "error": "not loaded yet"}


def load_model() -> bool:
    """Load the bundle. On failure keep the previously loaded model (if any)."""
    try:
        meta = json.loads((MODEL_DIR / "metadata.json").read_text())
        if meta.get("feature_cols") != FEATURE_COLS:
            raise ValueError("feature_cols in metadata.json do not match features.py (train/serve skew)")
        model = joblib.load(MODEL_DIR / "model.joblib")
    except Exception as exc:  # noqa: BLE001
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        MODEL_LOADED.set(1 if STATE["model"] is not None else 0)
        return False
    STATE.update(model=model, meta=meta, error=None)
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=str(meta["version"]), candidate=str(meta.get("candidate"))).set(1)
    MODEL_LOADED.set(1)
    return True


class Observation(BaseModel):
    timestamp: str
    requests: float = Field(ge=0)
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0)
    error_rate_pct: float = Field(ge=0)


class ScoreRequest(BaseModel):
    observations: list[Observation] = Field(min_length=5, max_length=500)


app = FastAPI(title="AIOps anomaly scoring service", version="1.0")
load_model()


@app.get("/health")
def health():
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "error": STATE["error"]})
    m = STATE["meta"]
    return {"status": "ok", "model_version": m["version"], "candidate": m.get("candidate"),
            "threshold": m["threshold"]}


@app.post("/reload")
def reload():
    if not load_model():
        raise HTTPException(status_code=500, detail=STATE["error"])
    return {"status": "reloaded", "model_version": STATE["meta"]["version"]}


@app.post("/score")
def score(req: ScoreRequest):
    model, meta = STATE["model"], STATE["meta"]
    if model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {STATE['error']}")
    start = time.perf_counter()
    df = pd.DataFrame([o.model_dump() for o in req.observations])
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing:
        raise HTTPException(status_code=400, detail="timestamps must be valid and in ascending order")
    feats = build_features(df)
    s = float(-model.score_samples(feats.iloc[[-1]])[0])
    is_anom = s >= float(meta["threshold"])
    INFER_SECONDS.observe(time.perf_counter() - start)
    SCORE.observe(s)
    PREDICTIONS.labels(result="anomaly" if is_anom else "normal").inc()
    last = req.observations[-1]
    for c in BASE_COLS:
        INPUT_VALUE.labels(metric=c).set(getattr(last, c))
    return {"timestamp": last.timestamp, "score": s, "threshold": meta["threshold"],
            "is_anomaly": bool(is_anom), "model_version": meta["version"]}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
