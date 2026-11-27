#!/usr/bin/env python3
"""Synthetic AIOps telemetry generator.

Produces per-minute service metrics with labelled incidents and
application logs with labelled incident periods. Fully seeded.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
TYPES = ["cpu_spike", "latency_spike", "error_burst", "memory_leak", "traffic_drop"]
VARIANTS = {
    "v1": {"seed": 42, "start": "2024-03-01", "req_scale": 1.0, "lat_scale": 1.0, "suffix": ""},
    "v2": {"seed": 7, "start": "2024-03-15", "req_scale": 1.25, "lat_scale": 1.4, "suffix": "_v2"},
}

NORMAL_LOGS = [
    (0.55, "INFO", "GET /api/orders/{oid} status=200 duration={ms}ms"),
    (0.20, "INFO", "User {uid} logged in from 10.0.{a}.{b}"),
    (0.15, "INFO", "Cache hit ratio {ratio}"),
    (0.10, "DEBUG", "DB query completed in {ms}ms rows={rows}"),
]
ANOMALY_LOGS = [
    ("ERROR", "Connection refused to db-{n}:5432"),
    ("WARN", "Retry {k}/5 for request {oid}"),
    ("ERROR", "Timeout after {ms}ms calling payment-svc"),
    ("ERROR", "OutOfMemoryError in worker-{n}"),
]


def make_metrics(rng, start, req_scale, lat_scale, days=14):
    idx = pd.date_range(start, periods=days * 1440, freq="min")
    n = len(idx)
    minute_of_day = np.arange(n) % 1440
    load = (np.sin(2 * np.pi * minute_of_day / 1440 - np.pi / 2) + 1) / 2
    weekend = np.isin(idx.dayofweek, [5, 6])
    requests = (300 + 500 * load * np.where(weekend, 0.6, 1.0)) * req_scale + rng.normal(0, 20, n)
    cpu = 15 + 0.07 * requests + rng.normal(0, 3, n)
    mem = 45 + rng.normal(0, 1.5, n)
    latency = (80 + 0.05 * requests) * lat_scale + rng.gamma(2.0, 3.0, n)
    error_rate = 0.5 + np.abs(rng.normal(0, 0.2, n))

    label = np.zeros(n, dtype=int)
    incidents = []
    for i in range(28):
        start_i = 360 + i * 700 + int(rng.integers(0, 300))
        kind = TYPES[i % len(TYPES)]
        dur = int(rng.integers(60, 121)) if kind == "memory_leak" else int(rng.integers(5, 21))
        sl = slice(start_i, start_i + dur)
        if kind == "cpu_spike":
            cpu[sl] += 35
        elif kind == "latency_spike":
            latency[sl] *= 4
        elif kind == "error_burst":
            error_rate[sl] += 8
        elif kind == "memory_leak":
            mem[sl] += np.linspace(10, 40, dur)
        elif kind == "traffic_drop":
            requests[sl] *= 0.2
            cpu[sl] = 15 + 0.07 * requests[sl] + rng.normal(0, 3, dur)
        label[sl] = 1
        incidents.append({
            "id": f"INC-{i + 1:03d}", "type": kind,
            "start": str(idx[start_i]), "end": str(idx[start_i + dur - 1]), "minutes": dur,
        })

    df = pd.DataFrame({
        "timestamp": idx,
        "requests": requests.round(1),
        "cpu_pct": np.clip(cpu, 0, 100).round(2),
        "mem_pct": np.clip(mem, 0, 100).round(2),
        "latency_ms": latency.round(1),
        "error_rate_pct": error_rate.round(3),
        "is_anomaly": label,
    })
    return df, incidents


def fill(tpl, rng):
    return tpl.format(
        oid=rng.integers(10000, 99999), ms=rng.integers(5, 400), uid=rng.integers(1000, 9999),
        a=rng.integers(0, 255), b=rng.integers(1, 255), ratio=f"{rng.uniform(0.7, 0.99):.2f}",
        rows=rng.integers(1, 500), n=rng.integers(1, 4), k=rng.integers(1, 6),
    )


def make_logs(rng, days=3):
    minutes = days * 1440
    base = datetime(2024, 3, 1)
    in_incident = np.zeros(minutes, dtype=bool)
    log_incidents = []
    for i in range(12):
        start = 120 + i * 350 + int(rng.integers(0, 150))
        dur = int(rng.integers(10, 26))
        in_incident[start:start + dur] = True
        log_incidents.append({"start_minute": start, "end_minute": start + dur - 1})

    weights = np.array([w for w, _, _ in NORMAL_LOGS])
    lines = []
    for m in range(minutes):
        picks = rng.choice(len(NORMAL_LOGS), size=rng.poisson(15), p=weights)
        events = [(NORMAL_LOGS[j][1], NORMAL_LOGS[j][2]) for j in picks]
        if rng.random() < 0.02:                      # benign background retries
            events.append(("WARN", "Retry {k}/5 for request {oid}"))
        if in_incident[m]:
            for _ in range(rng.poisson(8)):
                events.append(ANOMALY_LOGS[int(rng.integers(len(ANOMALY_LOGS)))])
        secs = np.sort(rng.integers(0, 60, len(events)))
        for s, (lvl, tpl) in zip(secs, events):
            ts = (base + timedelta(minutes=m, seconds=int(s))).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(f"{ts} app-0{rng.integers(1, 4)} {lvl} {fill(tpl, rng)}")
    return lines, log_incidents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="v1")
    args = ap.parse_args()
    v = VARIANTS[args.variant]
    rng = np.random.default_rng(v["seed"])
    DATA.mkdir(exist_ok=True)

    df, incidents = make_metrics(rng, v["start"], v["req_scale"], v["lat_scale"])
    df.to_csv(DATA / f"metrics{v['suffix']}.csv", index=False)
    (DATA / f"incidents{v['suffix']}.json").write_text(json.dumps(incidents, indent=2))
    print(f"metrics{v['suffix']}.csv: {len(df)} rows, {df.is_anomaly.sum()} anomalous minutes "
          f"({df.is_anomaly.mean():.1%}), {len(incidents)} incidents")

    if args.variant == "v1":
        lines, log_inc = make_logs(rng)
        (DATA / "app.log").write_text("\n".join(lines) + "\n")
        (DATA / "log_incidents.json").write_text(json.dumps(log_inc, indent=2))
        print(f"app.log: {len(lines)} lines, {len(log_inc)} log incident periods")


if __name__ == "__main__":
    main()
