#!/usr/bin/env python3
"""Generate synthetic service telemetry with incident labels."""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from common import FEATURES, TARGET, sha256_file


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(rows, seed, drifted):
    rng = np.random.default_rng(seed)
    cpu = np.clip(rng.normal(68 if drifted else 55, 15, rows), 0, 100)
    mem = np.clip(rng.normal(60, 12, rows), 0, 100)
    io_wait = np.clip(rng.gamma(2.0, 3.5 if drifted else 2.0, rows), 0, 60)
    latency = rng.lognormal(np.log(260 if drifted else 180), 0.35, rows)
    rps = np.clip(rng.normal(1100 if drifted else 800, 150, rows), 10, None)
    err = np.clip(rng.beta(1.2, 60, rows) * 100, 0, 100)
    gc = rng.gamma(2.0, 15.0, rows)

    z = (-10.0 + 0.045 * cpu + 0.035 * mem + 0.09 * io_wait + 0.006 * latency
         + 0.55 * err + 0.01 * gc + 1.5 * ((cpu > 75) & (io_wait > 8)))
    if drifted:
        # Concept drift: after the migration, request volume drives incidents.
        z = z + 0.01 * (rps - 1100)
    incident = (rng.random(rows) < sigmoid(z)).astype(int)

    df = pd.DataFrame({
        "cpu_util": cpu, "mem_util": mem, "io_wait": io_wait,
        "p95_latency_ms": latency, "req_per_sec": rps,
        "error_rate": err, "gc_pause_ms": gc,
    }).round(2)
    df[TARGET] = incident
    return df[FEATURES + [TARGET]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--drifted", action="store_true")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"telemetry_{args.version}.csv"
    df = generate(args.rows, args.seed, args.drifted)
    df.to_csv(path, index=False)

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    digest = sha256_file(path)
    manifest[args.version] = {
        "file": path.name, "rows": len(df), "seed": args.seed,
        "drifted": args.drifted, "incident_rate": round(float(df[TARGET].mean()), 4),
        "sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {path} rows={len(df)} incident_rate={df[TARGET].mean():.3f} "
          f"sha256={digest[:12]}")


if __name__ == "__main__":
    main()
