# ML Model Development and Deployment II: From Experiment Tracking to Monitored, Gated Model Releases on Kubernetes

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
The SRE group wanted an Incident Predictor: a model that reads one-minute service telemetry (CPU, memory, I/O wait, p95 latency, request rate, error rate, GC pause) and estimates the probability of an incident in the next 15 minutes. A notebook prototype existed. The job was to turn it into a repeatable, production-style lifecycle and run it entirely from the terminal.

## What I did
1. **Versioned the data:** generated four seeded datasets, fingerprinted them with SHA-256 in a manifest, and added a validation gate that exits non-zero on bad data.
2. **Tracked experiments:** ran a 6-run hyperparameter search (logistic regression, random forest, gradient boosting) in MLflow, logging parameters, data hashes, metrics, a tuned decision threshold, a signature and pinned requirements.
3. **Used the Model Registry:** registered the best run and promoted it with aliases (`champion`, `challenger`, `previous`) instead of hard-coded versions.
4. **Built the inference API:** FastAPI with schema validation, separate liveness and readiness endpoints, Prometheus metrics and 9 behavioural tests.
5. **Containerised it:** pinned dependencies, non-root user and a healthcheck.
6. **Deployed to Kubernetes (kind):** probes, resource limits, a security context and a zero-downtime rolling update, then diagnosed and rolled back two broken rollouts.
7. **Detected drift:** PSI and the KS test on a post-migration traffic window, then measured performance decay with delayed labels.
8. **Retrained and promoted:** trained a challenger on v1+v2 data, passed an automated champion/challenger gate, rolled out image 1.1.0 and practised a rollback.

## Results

### Data
| Dataset | Rows | Incident rate |
|---------|------|---------------|
| telemetry_v1 | 20,000 | 0.127 |
| telemetry_v1_live | 2,000 | 0.140 |
| telemetry_v2 (post-migration) | 8,000 | 0.341 |
| telemetry_v2_live | 2,000 | 0.354 |

All four files passed validation. A deliberately corrupted file (5 nulls in `error_rate`, 5 out-of-range `cpu_util` values) was rejected with exit code 1.

### Experiments (v1 data)
| Run | val PR AUC | val ROC AUC | Threshold |
|-----|-----------|-------------|-----------|
| logreg (C=0.1 / C=1.0) | 0.4617 | 0.8104 | 0.657 / 0.658 |
| random forest (depth 8 / 16) | 0.4282 / 0.4331 | 0.7961 / 0.7920 | 0.601 / 0.495 |
| hist gradient boosting | 0.4315 / 0.4342 | 0.7957 / 0.7927 | 0.246 / 0.242 |

Logistic regression won, so `v1-logreg-2` became champion v1 (threshold 0.6582). The v1 incident process is close to linear in log-odds, so a simple model is a legitimate winner. PR AUC was used for selection because only about 12% of samples are positive.

### Serving and container
- 9 of 9 API tests passed, and invalid input returned a structured 422.
- The image `incident-api:1.0.0` ran as uid 10001 and reported healthy (676 MB on disk, 155 MB of content; about 102 MiB RAM at idle).
- A wrong `MODEL_DIR` left Docker's healthcheck at "healthy" while `/readyz` returned 503, which is why Kubernetes needs a separate readiness probe.

### Kubernetes rollouts
- **Drill A, missing image:** a tag that was not on the node gave `ImagePullBackOff`. The old pods kept serving because `maxUnavailable: 0`, and `rollout undo` fixed it.
- **Drill B, bad config:** the new pod was Running but 0/1 Ready with zero restarts. Readiness blocked the rollout and liveness did not cause a restart loop.

### Drift
| Feature | PSI vs v1 training data | Status |
|---------|------------------------|--------|
| cpu_util | 0.7071 | significant |
| io_wait | 0.5376 | significant |
| p95_latency_ms | 1.1010 | significant |
| req_per_sec | 3.1484 | significant |
| mem_util, error_rate, gc_pause_ms | below 0.011 | stable |

The control window (same regime) showed no drift, and every PSI was below 0.011. The PSI self-test gave 0.0000 for identical samples and 0.9239 for a 1-sd shift. The champion's predicted incident rate jumped from 0.175 to 0.528 on the drifted window, an early symptom visible from `/metrics` without labels.

### Impact of drift on champion v1 (held-out test splits)
| Metric | v1 test | v2 test |
|--------|---------|---------|
| ROC AUC | 0.8294 | 0.7839 |
| Predicted rate | 0.1665 | 0.4975 |
| True rate | 0.1268 | 0.3406 |

ROC AUC does not depend on prevalence, so its drop of about 4.5 points is real ranking decay from concept drift (request volume now matters). PR AUC is not comparable across the two sets because their incident rates differ.

### Retrain, gate and promotion
| Metric | Champion v1 | Challenger v2 (hist_gb) |
|--------|-------------|-------------------------|
| v1 PR AUC | 0.5251 | 0.4980 |
| v2 PR AUC | 0.6703 | 0.7948 |
| v2 ROC AUC | 0.7839 | 0.8654 |
| v2 F1 | 0.6204 | 0.7042 |

- **Gate rules:** the challenger must gain at least 0.01 PR AUC on v2 data and lose no more than 0.05 on v1 data.
- **Outcome:** gain +0.1245 and v1 regression 0.0271, so the challenger passed.
- **Registry after promotion:** champion=2, previous=1.
- **Rollout and rollback:** image 1.1.0 was rolled out and all pods served model v2. A rollback to v1 and a roll-forward both worked.
- **After the rollout:** the predicted rate on the post-migration window was 0.504 against an actual 0.354. The v2 model still over-predicts a little because its threshold (0.286) is tuned for F1, which favours recall over precision.

### Validation
The lab's validation script passed all 20 checks.

## Problems I ran into
**1. MLflow would not start (the first full run failed 12 of 20 checks).** The server crashed with `cannot import name 'FallbackAsyncAdaptedQueuePool' from 'sqlalchemy.pool'`. MLflow 2.12.2 is incompatible with the SQLAlchemy 2.1.x that pip resolved. Everything downstream failed as a cascade: no registry, no exported model, no image, and pods stuck in `ImagePullBackOff`. I pinned `sqlalchemy==2.0.30` and `alembic==1.13.1`, deleted the broken MLflow database, added the pins to the requirements file and reran. All 20 checks then passed.

**2. Environment differed from the lab text.** The server had Python 3.12, Docker 29 and kubectl 1.36 instead of 3.10, 24 and 1.30. I added a fallback that imports the image into the kind node with `ctr` when `kind load` leaves it incomplete, and checked each image with `crictl` before deploying.

## What I learned
- Data is part of the model: seeded generation, SHA-256 fingerprints and a validation gate that exits non-zero make "which data trained this?" answerable.
- Choose metrics that fit the problem. PR AUC suits an imbalanced target, and the decision threshold is a tuned, versioned artefact.
- Across regimes with different prevalence, only prevalence-independent metrics such as ROC AUC are comparable.
- Aliases decouple deployment from versions, so promotion and rollback are registry operations, not code edits.
- Liveness and readiness must be separate probes, and `maxUnavailable: 0` kept the old pods serving through two bad rollouts.
- Output shifts show up in `/metrics` immediately, PSI and KS quantify input drift, and delayed labels confirm real performance decay.
- A promotion gate that compares both models on the same held-out sets, with a no-regression rule and a default of "reject", makes retraining safe to automate.
- Pin transitive dependencies too. One unpinned library broke the whole pipeline.

## Repo contents
- `reports/`: drift reports (baseline and current), the promotion report
- `data/manifest.json`: dataset fingerprints
- `ml/`, `serving/`, `k8s/`, `scripts/`: the code and manifests I wrote or completed

Course-provided starter files are not included.

## Tools
MLflow, scikit-learn, FastAPI, Prometheus client, pytest, Docker, kind, kubectl, SciPy, jq, Bash
