#!/usr/bin/env python3
"""Hourly traffic forecast: seasonal naive vs gradient boosting, plus P95 for capacity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import load_metrics

HORIZON = 96                 # forecast the last 4 days (hours)
REPLICA_CAPACITY = 250       # requests/min one replica serves at target utilisation
FEATS = ["hour", "hour_sin", "hour_cos", "dow", "is_weekend"]


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def seasonal_naive(series: pd.Series, season: int = 24) -> pd.Series:
    return series.shift(season)


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow"] = idx.dayofweek
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/metrics.csv")
    args = ap.parse_args()

    df = load_metrics(args.data)
    y = df.set_index("timestamp")["requests"].resample("1h").mean()
    X = calendar_features(y.index)
    X_tr, y_tr, X_te, y_te = X.iloc[:-HORIZON], y.iloc[:-HORIZON], X.iloc[-HORIZON:], y.iloc[-HORIZON:]

    naive = seasonal_naive(y).iloc[-HORIZON:]
    gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
    pred = pd.Series(gbm.fit(X_tr[FEATS], y_tr).predict(X_te[FEATS]), index=y_te.index)
    q95m = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=300,
                                         learning_rate=0.05, random_state=42)
    p95 = pd.Series(q95m.fit(X_tr[FEATS], y_tr).predict(X_te[FEATS]), index=y_te.index)

    res = {
        "naive": {"mae": round(float((y_te - naive).abs().mean()), 2), "mape": round(mape(y_te, naive), 2)},
        "gbm": {"mae": round(float((y_te - pred).abs().mean()), 2), "mape": round(mape(y_te, pred), 2)},
        "p95_coverage": round(float((y_te <= p95).mean()), 3),
    }
    err = pd.DataFrame({"naive_abs": (y_te - naive).abs(), "gbm_abs": (y_te - pred).abs(), "p95": p95})
    daily = err.groupby(err.index.date).agg(naive_mae=("naive_abs", "mean"),
                                            gbm_mae=("gbm_abs", "mean"), peak_p95=("p95", "max"))
    daily["replicas"] = np.ceil(daily["peak_p95"] / REPLICA_CAPACITY).astype(int)
    daily.index = daily.index.astype(str)
    daily.index.name = "date"

    print(f"Test horizon: {y_te.index[0]} -> {y_te.index[-1]} ({HORIZON} hours)")
    print(f"Seasonal naive : MAE={res['naive']['mae']:7.2f}  MAPE={res['naive']['mape']:5.2f}%")
    print(f"Gradient boost : MAE={res['gbm']['mae']:7.2f}  MAPE={res['gbm']['mape']:5.2f}%")
    print(f"P95 coverage   : {res['p95_coverage']:.1%} of actual hours below the P95 forecast\n")
    print("Per-day error and capacity plan:")
    print(daily.round(1).to_string())
    res["capacity_plan"] = daily.round(2).reset_index().to_dict("records")
    Path("results").mkdir(exist_ok=True)
    Path("results/forecast.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
