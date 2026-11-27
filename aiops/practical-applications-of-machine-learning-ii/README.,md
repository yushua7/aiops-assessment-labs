# Practical Applications of Machine Learning II: AIOps Anomaly Detection, Forecasting and Model Operations

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
The SRE team for an e-commerce platform was tired of static alert thresholds that fire constantly at peak traffic and miss slow failures such as memory leaks. The job was to build an ML operations pipeline that detects anomalies in metrics and logs, forecasts traffic for capacity planning, serves a tracked and registered model as a containerised API, monitors it for drift, and retrains and promotes a new model safely. All data came from a seeded local simulator that injects labelled incidents.

## What I did
1. **Generated telemetry:** 14 days of per-minute metrics with 28 labelled incidents (5 types), 3 days of logs (66,584 lines) and a drifted "post-release" v2 dataset.
2. **Built a statistical baseline:** a rolling robust z-score detector, evaluated with event recall and false alerts per day, not just F1. A shared feature module was used for both training and serving.
3. **Trained ML detectors:** Isolation Forest and Local Outlier Factor sweeps tracked in MLflow, with the threshold tuned on validation data only and the best model registered as `champion`.
4. **Mined logs:** Drain3 template mining and an Isolation Forest on per-window template counts.
5. **Forecast traffic:** seasonal naive against gradient boosting, plus a P95 forecast turned into a replica count.
6. **Served the model:** a FastAPI scoring service in Docker with input validation, Prometheus metrics, a read-only model mount and an offline/online consistency check.
7. **Monitored drift:** Prometheus recording and alert rules, PSI and KS drift tests, and a replay of drifted traffic to show the operational impact.
8. **Retrained and promoted:** a champion/challenger gate, hot reload of the running service and a rollback drill.

## Results

### Data
| Item | Value |
|------|-------|
| Metrics (v1) | 20,160 minutes, 677 anomalous (3.4%), 28 incidents |
| Incident types | cpu_spike 6, error_burst 6, latency_spike 6, memory_leak 5, traffic_drop 5 |
| Logs | 66,584 lines, 12 incident periods |
| Metrics (v2, post-release) | 20,160 minutes, 782 anomalous (3.9%), 28 incidents |

### Statistical baseline (test split, days 11-14)
| Threshold | Precision | Recall | F1 | Event recall | False alerts/day |
|-----------|-----------|--------|----|--------------|------------------|
| 3.0 | 0.252 | 0.741 | 0.376 | 1.0 | 82.75 |
| 4.0 | 0.610 | 0.717 | 0.659 | 1.0 | 18.75 |
| 5.0 | 0.861 | 0.711 | 0.779 | 1.0 | 4.75 |
| 6.0 | 0.958 | 0.687 | 0.800 | 1.0 | 1.25 |

Raising the threshold cuts false alerts sharply at a small cost in recall. That is the alert-fatigue trade-off. The most-deviating metric for each incident matched its type, which makes a usable root-cause hint.

### ML detectors (MLflow sweep)
| Candidate | Val F1 | Test F1 | Test false alerts/day |
|-----------|--------|---------|-----------------------|
| iforest n100 s256 | 0.580 | 0.446 | 14.0 |
| iforest n100 s1024 | 0.599 | 0.558 | 6.0 |
| iforest n200 s256 | 0.554 | 0.438 | 10.0 |
| iforest n200 s1024 | **0.607** | 0.636 | 16.5 |
| LOF k=20 | 0.316 | 0.381 | 5.0 |
| LOF k=50 | 0.556 | 0.584 | 4.2 |

`iforest_n200_s1024` won on validation F1 and became champion v1 (threshold 0.4866). **This did not beat the baseline.** Its best test F1 (0.636) is below the baseline's 0.78 to 0.80 at thresholds 5 and 6, and it raised 16.5 false alerts per day against 1.25 to 4.75 for the baseline at those thresholds. Selecting on validation F1 picked a model that was not the best on test either (LOF k=50 had 4.2 false alerts per day). A tuned statistical baseline is a hard bar for unsupervised ML to clear on this data.

### Log anomaly detection
Drain3 reduced 66,584 lines to 8 templates (for example `ERROR Timeout after <NUM>ms calling payment-svc`). An Isolation Forest on per-window template counts reached test F1 0.954 with event recall 1.0 and 0 false alerts per day. The most anomalous window had the timeout, connection-refused and retry templates at 22.6x, 20.9x and 12.3x their normal counts, which points straight at the cause.

### Forecasting
| Model | MAE | MAPE |
|-------|-----|------|
| Seasonal naive | 29.06 | 4.68% |
| Gradient boosting | 4.13 | 0.81% |

Seasonal naive failed on Monday (MAE 103.7 against 3.3), because it copies Sunday's low weekend traffic. P95 coverage was 86.5%, below the 95% target, so the capacity plan may under-provision some hours. It recommended 4 replicas each day.

### Serving
- The service returned 503 with a clear error when started without the model, then healthy once the model was mounted.
- Invalid inputs were rejected cleanly: missing field, too few observations and out-of-order timestamps (400).
- **Train/serve consistency:** offline and online scores matched exactly (max difference 0.0) on 4 sample rows.
- The image is 739 MB on disk (169 MB content).

### Drift and operational impact
| Metric | PSI | Status |
|--------|-----|--------|
| requests | 1.1411 | significant |
| cpu_pct | 0.4255 | significant |
| latency_ms | 8.2566 | significant |
| mem_pct, error_rate_pct | below 0.008 | none |

The v1-against-v1 sanity check showed no drift (all PSI below 0.05). Replaying drifted traffic through the old champion raised the alert rate from 2.9% to 13.6% and dropped precision from 0.59 to 0.25, with recall of 1.0. The `ModelAnomalyRateHigh` rule was loaded, but I did not capture its firing state in my output.

### Retrain and promotion
| Model (on v2 test data) | F1 | Precision | Recall | False alerts/day | Alert rate |
|-------------------------|----|-----------|--------|------------------|------------|
| Champion v1 | 0.395 | 0.253 | 0.899 | 108.0 | 12.8% |
| Challenger v2 (iforest n100 s1024) | 0.624 | 0.590 | 0.663 | 22.5 | 4.1% |

The challenger gained 0.229 F1, well over the 0.02 gate, so it was promoted (champion=v2, previous=v1). After a hot reload the service reported version 2 with no rebuild and no dropped requests. Replaying v2 traffic gave an alert rate of 4.8% with precision 0.65 and recall 0.90. The rollback drill exported `previous` (v1) cleanly into a scratch directory.

### Validation
The lab's validation script passed all 19 checks.

## Problems I ran into
**1. Dependency conflict made the first full run fail (4 of 19 checks passed).** `mlflow 2.14.3` needs `cachetools>=5,<6`, but `drain3 0.9.11` pins `cachetools==4.2.1`, so pip refused to resolve the requirements file. Because the venv install failed, every later step failed with `ModuleNotFoundError`. My retry loop and the fallback pins I had added did not fix it. I removed `drain3` from the requirements file and installed it separately with `--no-deps`, plus `jsonpickle>=3,<4` and `cachetools>=5,<6`. I reran everything from a clean venv and all 19 checks passed.

**2. Environment differed from the lab text.** The server had Python 3.12 and Docker 29, not 3.10 and 24. The container still used `python:3.10-slim`, and the pinned packages worked in both.

**3. Prometheus checks looked empty in the first run.** The service had no model, so scoring never happened and the counters stayed at 0. That was a symptom of problem 1, not a Prometheus problem.

## What I learned
- Judge anomaly detectors by event recall and false alerts per day, and always against a tuned statistical baseline. Here the baseline was better than the ML model.
- Choosing a model by validation F1 does not guarantee it is best on test or in operations. The threshold choice is part of the model.
- Template mining turns noisy logs into a handful of event types, and count windows give both detection and an explanation.
- Calendar-aware forecasts handle regime changes (Monday after a weekend) that naive copies cannot, and P95 coverage must be checked before trusting a capacity plan.
- One shared feature module plus an offline/online score comparison guards against train/serve skew.
- Drift looks like an alert storm: a high anomaly rate with collapsing precision means the inputs changed. PSI shows which features moved.
- A champion/challenger gate on identical held-out data, with aliases and hot reload, makes promotion and rollback safe and cheap.
- Pin transitive dependencies, and install the one conflicting package separately instead of letting the resolver fail the whole install.

## Repo contents
- `results/`: baseline sweep, template list, log anomaly, forecast, drift and promotion reports
- `src/`: simulator, features, evaluation, detectors, log, forecast, drift and promotion scripts
- `service/`: the FastAPI app, Dockerfile and requirements
- `monitoring/`: Prometheus config and alert rules

Course-provided starter files are not included. The MLflow database, trained model bundles and generated data are not included.

## Tools
Python, scikit-learn, MLflow, Drain3, FastAPI, Prometheus, Docker, SciPy, jq, Bash
