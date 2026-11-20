# Deep Learning, Open Source Perspective: From PyTorch Training to a Containerised ONNX Anomaly Detector

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
An SRE team wants a deep-learning anomaly detector for service telemetry (CPU, memory, latency, error rate, one sample per minute), under a strict policy: only open-source components, reproducible and tracked training, a framework-neutral production artefact, and Prometheus metrics. I built the whole lifecycle on a CPU-only server.

## What I did
1. **Reproducible stack:** installed a CPU-only PyTorch build with pinned versions, saved a lock file (74 pinned packages), and audited licences.
2. **Data pipeline:** generated 14 days of telemetry with 12 labelled incidents (all in the second week), split it chronologically, fitted the scaler on training data only, and built sliding windows.
3. **Autoencoder:** trained a PyTorch window autoencoder with fixed seeds and tracked the run in MLflow.
4. **Threshold and evaluation:** calibrated the alert threshold on the validation week and evaluated on the test week, including recall per incident type.
5. **ONNX:** exported to ONNX with a dynamic batch axis, validated the graph, checked parity with PyTorch, and benchmarked both.
6. **MLflow registry:** ran an MLflow tracking server, trained a second model with a different bottleneck, compared the runs through the REST API, and promoted the better one with a `champion` alias.
7. **Container:** packaged the champion in a FastAPI and ONNX Runtime container with no PyTorch, a non-root user, a read-only filesystem, resource limits, and Prometheus metrics.
8. **Serving check:** replayed the whole test week through the API and compared every score with the offline results, then load-tested it.

## Results

### Stack and data
- Packages used were permissive: Apache, BSD, or MIT licences, and no GPL-family licences were found in the environment.
- 20,160 rows, with 1,350 anomalous minutes (cpu_spike 270, error_burst 210, latency_degradation 360, memory_leak 510).
- Windows of 32 minutes: 3,585 train, 1,425 validation, 10,049 test, of which 1,722 test windows touch an incident.
- Thread scaling: 20 large matrix multiplications took 0.33 s with 1 thread and 0.34 s with 2, so on this machine a second thread gave no speedup.

### Autoencoder (18,704 parameters, 25 epochs in 7.3 s)
Final validation loss was 0.1675. On the test week, with the threshold at the 99.5th percentile of validation error (0.2715):

| TP | FP | FN | TN | Precision | Recall | F1 | False-positive rate |
|----|----|----|----|-----------|--------|----|---------------------|
| 1,625 | 28 | 97 | 8,299 | 0.983 | 0.944 | 0.963 | 0.0034 |

| Incident type | Recall |
|---------------|--------|
| error_burst | 1.000 |
| latency_degradation | 1.000 |
| cpu_spike | 0.978 |
| memory_leak | 0.852 |

The memory leak is the weakest, because it ramps up slowly from zero, so its earliest windows look normal. Lowering the threshold percentile to 99.0 gave a threshold of 0.2528, with precision 0.961, recall 0.948, and F1 0.954. Recall barely moved while the false-positive rate more than doubled (0.0034 to 0.0079), so the stricter 99.5 setting was the better choice here.

### Two architectures compared
| Model | Bottleneck | Parameters | Validation loss | Precision | Recall | F1 |
|-------|-----------|-----------|-----------------|-----------|--------|----|
| Run A | 16 | 18,704 | 0.168 | 0.983 | 0.944 | 0.963 |
| Run B | 4 | 17,156 | 0.195 | 0.978 | 0.938 | 0.957 |

Run A was promoted to `champion` automatically, based on F1 read from the MLflow REST API. The difference between the two is small (0.006 F1) and comes from one run each, so it shouldn't be read as a firm ranking.

### ONNX export
- The exported graph (74.6 KiB, opset 17) contained only `Flatten`, `Gemm`, `Relu`, `Reshape`, and `Constant` operations.
- Outputs matched PyTorch exactly (maximum difference 0.0).
- The dynamic batch axis worked for batches of 1, 7, and 300, and a wrong input shape was rejected.

| | PyTorch | ONNX Runtime |
|--|---------|--------------|
| One window | 0.078 ms | 0.016 ms (about 5 times faster) |
| 256 windows | 0.210 ms | 0.120 ms (about 1.8 times faster) |

### Container and serving
- The image is 524 MB, has no PyTorch installed, runs as user 10001, and reports healthy.
- A normal window scored 0.144 (threshold 0.271) and was not flagged. A CPU spike scored 1.51 and was flagged with `cpu_pct` as the top feature. A latency degradation scored 30.6 and was flagged with `latency_ms` as the top feature. A window with 31 rows instead of 32 returned HTTP 422.
- **Replay of the whole test week (10,049 windows):** 100% of the flag decisions matched the offline results, the service F1 was 0.963 (identical), and the largest score difference was 3.7e-4. That difference is larger than the lab text's example, but it is within the lab's tolerance (1e-3) and flipped no decisions.
- **Load test (2,000 requests, 8 concurrent):** 0 failures, 635 requests per second, p95 latency 18 ms and p99 22 ms. The mean ONNX inference took 0.5 ms, and the container used 51% CPU and 57 MiB of memory.

### Validation
The lab's validation script passed all 18 checks.

## A problem I ran into
When I re-evaluated the model at the 99.0 percentile, MLflow rejected the run because a parameter can only be logged once (`threshold_percentile` was already stored as 99.5). The evaluation printed its results but the logging step failed. I re-ran the evaluation at 99.5 to restore the deployed calibration. The lesson is that MLflow parameters are immutable per run, so values that change between evaluations should be logged as metrics, or each evaluation should get its own run.

## What I learned
- A reproducible ML environment needs a pinned CPU-only build, a lock file, fixed seeds, and a licence audit.
- Fitting the scaler on training data only, and verifying window arithmetic (1,350 anomalous minutes plus 12 incidents times 31 edge windows gives 1,722), prevents leakage.
- A threshold percentile trades false alarms against missed incidents, and per-type recall shows where a model is blind.
- ONNX export has to be proven with a parity check, and it removed PyTorch from the serving image.
- A replay of real traffic through the service is what proves the deployed artefact matches the offline model.
- The data is synthetic, so the high F1 shows that the pipeline works, not how this detector would do on real telemetry.

## Repo contents
- `results/`: threshold calibration for both runs, the pinned dependency lock file, the licence report, and the scaler

Course-provided scripts and data generators are not included.

## Tools
Python, PyTorch, ONNX, ONNX Runtime, MLflow, scikit-learn, FastAPI, Prometheus client, Docker
