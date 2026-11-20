# Deep Learning II: Sequence Models, Autoencoders and Attention for AIOps Anomaly Detection

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
An online payments provider has noisy threshold-based alerts that miss subtle incidents. The goal was to build deep-learning detectors for service metrics (CPU, memory, latency, error rate) and for application logs, compare them, and ship one as a scoring service. Everything ran on a CPU-only server using generated data: 14 days of 1-minute metrics with 6 labelled incidents, and 35,691 log lines from 3,000 sessions (240 anomalous).

## What I did
1. **Log parsing:** parsed raw log lines into per-session event-ID sequences with anchored regular expressions. All 35,691 lines matched, and the sequences were identical to the ground truth.
2. **Leakage-safe pipeline:** split the metrics chronologically (train 14,112 rows, validation 2,016, test 4,032), fitted the scaler on training data only, and built sliding windows. All 200 anomalous points sit in the test split.
3. **Forecasters:** trained LSTM and GRU models to predict the next 5 minutes of latency from the previous 60, with gradient clipping, dropout, weight decay, learning-rate scheduling, and early stopping, and compared them with a persistence baseline.
4. **LSTM autoencoder:** trained an unsupervised detector on normal windows, set the threshold from validation data, and evaluated it per incident, not only per point.
5. **DeepLog-style model:** trained a next-event LSTM on normal log sessions and tuned the top-k decision rule on validation data.
6. **Transformer classifier:** built a Transformer encoder with a padding mask and masked pooling, with class weights for the 8% anomalous sessions, and tested that padding doesn't change the output.
7. **Serving:** exported the autoencoder to ONNX, checked parity with PyTorch, and served it from a non-root FastAPI container.

## Results

### Latency forecasting (clean test windows, 3,384)
| Model | Parameters | MAE | Improvement over persistence |
|-------|-----------|-----|------------------------------|
| Persistence baseline | 0 | 5.41 ms | n/a |
| LSTM | 51,525 | 3.85 ms | about 29% |
| GRU | 38,725 | 3.83 ms | about 29% |

Both networks clearly beat the baseline, with errors that stay about the same from t+1 to t+5 (latency here is mostly slow seasonality plus noise). The 0.02 ms gap between the LSTM and GRU is too small to mean anything, but the GRU reaches it with about a quarter fewer parameters.

### LSTM autoencoder (11,924 parameters)
The model caught **6 of 6** incidents. Detection delay was 0 minutes for five of them and 9 minutes for the gradual memory leak. Point-wise precision was 0.589, recall 0.895, and F1 0.710 at the 99.5th-percentile threshold (0.178).

Changing the validation percentile used for the threshold barely changed detection but changed the noise a lot:

| Threshold percentile | Precision | Recall | F1 | Incidents caught | False-alarm episodes |
|----------------------|-----------|--------|----|------------------|----------------------|
| 99.0 | 0.560 | 0.905 | 0.692 | 6/6 | 21 |
| 99.5 | 0.589 | 0.895 | 0.710 | 6/6 | 11 |
| 99.9 | 0.633 | 0.870 | 0.733 | 6/6 | 3 |

On this data the stricter threshold looks better for an on-call team: every incident was still caught, with far fewer false alarms. The two hardest incidents (a small latency shift and a CPU/latency decoupling) scored only about 1.2 to 1.5, well above the threshold but far below the spikes (up to 45), so they are caught by the model learning the relationship between metrics and not by any single metric crossing a line.

### Log anomaly detection (test sessions)
| Model | Labels needed | Parameters | Precision | Recall | F1 |
|-------|---------------|-----------|-----------|--------|----|
| DeepLog-style LSTM (k=5) | No | 55,406 | 1.000 | 1.000 | 1.000 |
| Transformer encoder | Yes | 19,138 | 1.000 | 1.000 | 1.000 |

On validation data, k=5 was the best setting (F1 1.0), while k=4 gave 0.793 and k=6 gave 0.977. The Transformer's output was identical (difference 0.0) with and without padding, which shows the mask works.

**Caveat:** both models scored perfectly on every anomaly type, including out-of-order events, which the lab expected to be harder. The logs are synthetic and the anomalies are cleanly separable, so this result can't separate the two approaches or say how they'd do on real logs.

### ONNX export and serving
- The ONNX output matched PyTorch to within 6e-7 (reconstruction) and 2e-6 (score).
- Scoring a batch of 512 windows with onnxruntime took 60.8 ms.
- The container answered a normal window as not anomalous (score 0.065 against a threshold of 0.178), flagged a latency spike (score 20.3, top feature `latency_ms`), and flagged an error burst (score 45.0, top feature `error_rate`).
- A malformed window with 59 rows instead of 60 returned HTTP 422 with a message stating the expected shape.
- Over 100 requests, latency was p50 2.8 ms and p95 3.0 ms, using about 51 MiB of memory.
- The Docker image uses 442 MB of disk (100 MB of content).

### Validation
The lab's validation script passed all 10 checks.

## What I learned
- Chronological splits, scaling fitted on training data only, and correctly aligned windows come before any model choice.
- A forecaster is only worth deploying if it clearly beats the "same as now" baseline, which these did.
- For anomaly detection, the threshold and the incident-level view (delay and false-alarm episodes) matter as much as the network, as the 21, 11 and 3 false-alarm episodes show.
- A padding mask has to be tested, since without it predictions change with how much padding a session has.
- Perfect scores on synthetic data tell you the pipeline works, not how it will do on real data.

## Repo contents
- `results/`: forecasting, autoencoder, DeepLog, Transformer, and ONNX-parity results (JSON)

Course-provided scripts and the data generator are not included.

## Tools
Python, PyTorch, scikit-learn, pandas, NumPy, ONNX, ONNX Runtime, FastAPI, Docker
