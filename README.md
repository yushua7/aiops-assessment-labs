# Advanced ML for AIOps: Anomaly Detection, Explainability and Model Serving

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
An industrial machine reports its internal temperature every 5 minutes and has failed several times. The goal was to build a trustworthy detector, explain why it raises alarms, track every experiment, and serve it as an HTTP scoring service.

## Data
- [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB): `machine_temperature_system_failure` series, with labelled anomaly windows
- [LogHub](https://github.com/logpai/loghub): BGL_2k supercomputer logs with alert labels

Datasets are not included in this repo.

## What I did
1. Turned labelled anomaly windows into per-point labels and decomposed the signal with STL.
2. Wrote the feature engineering (lags, differences, trailing rolling statistics, z-score, time features) and trained XGBoost and LightGBM with a strict chronological train/test split to avoid leakage.
3. Tuned LightGBM with Optuna (30 trials) and tracked every run in an MLflow server, then registered the best model with a `champion` alias.
4. Explained the model with SHAP.
5. Built an unsupervised PyTorch autoencoder and compared it with Isolation Forest, using both point metrics and event metrics.
6. Wrote the log-line feature code and trained a classifier on LogHub BGL, then tested it for label leakage.
7. Served the registered model over REST with `mlflow models serve` and scored live rows with curl.

## Results

### Supervised models (test set, about 10.6% positive)
| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| XGBoost baseline | 0.833 | 0.240 |
| LightGBM baseline | 0.860 | 0.277 |
| LightGBM tuned (Optuna) | 0.865 | 0.299 |

PR-AUC is the main metric because the data is imbalanced (random is about 0.106). Tuned validation PR-AUC was 0.619 but test PR-AUC was 0.299. That gap shows selection overfitting when there are only a few incidents. At the default 0.5 threshold the baselines flagged nothing (F1 = 0), which shows how much the threshold matters.

### Unsupervised detectors (test month)
| Detector | ROC-AUC | PR-AUC | Incident detected | False-alarm episodes |
|----------|---------|--------|-------------------|----------------------|
| PyTorch autoencoder | 0.883 | 0.326 | Yes, 2014-02-09 12:05 | 4 (0.21/day) |
| Isolation Forest | 0.991 | 0.875 | Yes, 2014-02-07 18:50 | 13 (0.70/day) |

Isolation Forest scored far higher and caught the incident earlier but raised about three times as many false-alarm episodes. The autoencoder was quieter but detected the incident late. The test month contains only one incident, so these numbers show a trade-off, not a general ranking.

### Log anomaly classifier (5-fold CV on 2,000 lines)
| Features | PR-AUC | F1 |
|----------|--------|----|
| Message content + metadata (Level, Component) | 0.996 | 0.976 |
| Message content only | 0.995 | 0.976 |

Removing the metadata barely changed the score, so the message text carries the real signal. The sample is small and uses shuffled folds, so real use would need a time-based split.

## Key findings (from SHAP)
- Long-window (24 h) features carry about 46% of total importance and short-window features about 26%. The failures look like gradual drift, not sudden spikes.
- `hour` and `dayofweek` together carry about 25%. That is a risk, because the model could misfire if the operating schedule changes.
- SHAP values add up exactly to the model output, so every alert can be explained as a sum of feature effects.

Full written answers are in [shap_answers.md](shap_answers.md).

## Repo contents
- `src/features.py`: time-series feature engineering
- `src/logs_clf.py`: log classifier features and evaluation
- `results/`: metrics (JSON) and plots (EDA, SHAP, autoencoder error)
- `requirements.txt`: package versions used

Course-provided scripts and datasets are not included.

## Tools
Python, pandas, statsmodels, XGBoost, LightGBM, scikit-learn, Optuna, MLflow, SHAP, PyTorch
