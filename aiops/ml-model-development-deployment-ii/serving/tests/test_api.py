from fastapi.testclient import TestClient

from app.main import STATE, app

client = TestClient(app)

HIGH_RISK = {"cpu_util": 92, "mem_util": 85, "io_wait": 14, "p95_latency_ms": 520,
             "req_per_sec": 900, "error_rate": 7, "gc_pause_ms": 70}
LOW_RISK = {"cpu_util": 20, "mem_util": 30, "io_wait": 0.5, "p95_latency_ms": 90,
            "req_per_sec": 500, "error_rate": 0.1, "gc_pause_ms": 5}


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_readyz_reports_loaded_model():
    r = client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["model_version"] == str(STATE.meta["model_version"])


def test_model_metadata_contract():
    body = client.get("/model").json()
    assert 0 < body["threshold"] < 1
    assert body["features"] == list(HIGH_RISK.keys())


def test_predict_batch_shape_and_ranges():
    r = client.post("/predict", json={"instances": [HIGH_RISK, LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    assert all(0.0 <= p["incident_probability"] <= 1.0 for p in preds)


def test_behaviour_high_vs_low_risk():
    preds = client.post("/predict", json={"instances": [HIGH_RISK, LOW_RISK]}).json()["predictions"]
    assert preds[0]["incident"] is True
    assert preds[1]["incident"] is False
    assert preds[0]["incident_probability"] > preds[1]["incident_probability"]


def test_rejects_out_of_range_value():
    r = client.post("/predict", json={"instances": [{**HIGH_RISK, "cpu_util": 150}]})
    assert r.status_code == 422


def test_rejects_missing_field():
    bad = {k: v for k, v in HIGH_RISK.items() if k != "error_rate"}
    assert client.post("/predict", json={"instances": [bad]}).status_code == 422


def test_rejects_empty_batch():
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_metrics_exposed():
    client.post("/predict", json={"instances": [LOW_RISK]})
    text = client.get("/metrics").text
    assert "incident_predictions_total" in text
    assert "incident_model_info" in text
