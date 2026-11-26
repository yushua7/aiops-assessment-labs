#!/usr/bin/env python3
"""Generate a reproducible synthetic corpus of AIOps incident tickets.

Each record: id, split, label (resolver category), text and character-offset
entity annotations for HOST, IP, SERVICE and ERROR_CODE.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 42
N_PER_CLASS = 300
NOISE_RATE = 0.03
OUT = Path("data")
rng = random.Random(SEED)

TEMPLATES = {
    "database": [
        "Deadlock detected on {db} database on {host}, error {code} raised by {svc}",
        "Replication lag on {host} is {num} seconds for {db}",
        "{svc} cannot acquire a connection from the pool, {db} reports {code}",
        "Slow queries on {db} at {host} ({ip}), average latency {num} ms",
        "Database {db} on {host} rejected new connections with {code}",
        "Transaction rollback storm in {db}, {svc} logging {code}",
        "{svc} timing out while talking to {db} on {host}",
    ],
    "network": [
        "Packet loss of {pct} percent between {host} and {ip}",
        "{svc} on {host} fails to reach {ip} with {code}",
        "DNS lookups from {host} timing out, upstream resolver {ip} not answering",
        "BGP session flapping on edge router {host}, peer {ip}",
        "High latency on link to {ip}, {svc} reports {code}",
        "Load balancer health checks failing for {host}, {code} from {ip}",
        "{svc} timing out while talking to {host} ({ip})",
    ],
    "storage": [
        "Disk usage on {host} reached {pct} percent on {path}",
        "{svc} write failed on {host} with {code}, volume {path} is read-only",
        "NFS mount {path} on {host} is unresponsive, error {code}",
        "RAID array degraded on {host}, one disk reporting {code}",
        "Backup to {ip} failed because {path} has no space left, {code}",
        "I/O wait above {pct} percent on {host}, storage latency spiking",
    ],
    "application": [
        "{svc} returning {code} to users after deploy on {host}",
        "{svc} pods restarting in a loop on {host}, exit code {code}",
        "Memory usage of {svc} on {host} keeps growing until it is killed with {code}",
        "Unhandled exception in {svc}, stack trace shows {code}",
        "Response time of {svc} above {num} ms, error rate rising with {code}",
        "Feature flag change broke checkout flow in {svc} on {host}",
        "{svc} timing out for users, {code} seen on {host}",
    ],
    "security": [
        "Multiple failed SSH logins on {host} from {ip}, alert {code}",
        "WAF blocked SQL injection attempt against {svc} from {ip}, rule {code}",
        "Suspicious outbound traffic from {host} to {ip} flagged {code}",
        "User {user} attempted privilege escalation on {host}, {code}",
        "Expired TLS certificate on {svc} detected by scanner, {code}",
        "Malware signature found on {host}, quarantine triggered with {code}",
    ],
}
CODES = {
    "database": ["ORA-00060", "PG-53300", "MYSQL-1205", "PG-40P01", "ORA-01555"],
    "network": ["ETIMEDOUT", "ECONNRESET", "EHOSTUNREACH", "NET-0042"],
    "storage": ["ENOSPC", "EIO", "XFS-117", "NFS-STALE"],
    "application": ["HTTP-500", "HTTP-503", "OOM-137", "NPE-001"],
    "security": ["SEC-4011", "AUTH-401", "WAF-942100", "IDS-2019"],
}
HOST_PREFIX = ["db", "web", "app", "edge", "stor", "auth", "mq", "k8s-node", "cache", "lb"]
HOST_SUFFIX = ["", "", ".prod.local", ".dc1.internal"]
SERVICES = ["payments-api", "auth-service", "inventory-svc", "checkout-web",
            "orders-worker", "search-api", "billing-batch", "notification-svc"]
DATABASES = ["orders", "customers", "ledger", "inventory", "sessions"]
PATHS = ["/var/log", "/data", "/mnt/backups", "/var/lib/docker", "/srv/nfs/shared"]
USERS = ["jsmith", "adeleon", "svc_backup", "mkhan"]
PREFIXES = ["", "", "", "URGENT: ", "[P1] ", "[P2] ", "[P3] ",
            "Customer reports: ", "Monitoring alert - "]
SUFFIXES = ["", "", " Please investigate.", " Started after last deploy.",
            " Happening intermittently.", " Raised by on-call."]

PH = re.compile(r"\{(\w+)\}")
ENT_MAP = {"host": "HOST", "ip": "IP", "svc": "SERVICE", "code": "ERROR_CODE"}


def make_values(label):
    return {
        "host": f"{rng.choice(HOST_PREFIX)}-{rng.randint(1, 40):02d}{rng.choice(HOST_SUFFIX)}",
        "ip": f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "svc": rng.choice(SERVICES),
        "code": rng.choice(CODES[label]),
        "num": str(rng.randint(30, 2000)),
        "pct": str(rng.randint(20, 99)),
        "db": rng.choice(DATABASES),
        "path": rng.choice(PATHS),
        "user": rng.choice(USERS),
    }


def render(template, values):
    """Fill placeholders and record [start, end, LABEL] for entity placeholders."""
    out, spans, pos = "", [], 0
    for m in PH.finditer(template):
        out += template[pos:m.start()]
        key = m.group(1)
        start = len(out)
        out += values[key]
        if key in ENT_MAP:
            spans.append([start, len(out), ENT_MAP[key]])
        pos = m.end()
    out += template[pos:]
    return out, spans


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    for label, templates in TEMPLATES.items():
        for _ in range(N_PER_CLASS):
            body, spans = render(rng.choice(templates), make_values(label))
            prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
            text = prefix + body + suffix
            spans = [[s + len(prefix), e + len(prefix), lab] for s, e, lab in spans]
            if rng.random() < 0.10:          # lower-casing keeps offsets valid
                text = text.lower()
            records.append({"text": text, "label": label, "entities": spans})

    rng.shuffle(records)
    n = len(records)
    n_train, n_val = n * 70 // 100, n * 15 // 100
    labels = sorted(TEMPLATES)
    noisy = 0
    for i, rec in enumerate(records):
        rec["id"] = f"INC{100000 + i}"
        rec["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        if rec["split"] == "train" and rng.random() < NOISE_RATE:
            rec["label"] = rng.choice([l for l in labels if l != rec["label"]])
            noisy += 1

    with open(OUT / "tickets_all.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps({k: rec[k] for k in ("id", "split", "label", "text", "entities")}) + "\n")
    df = pd.DataFrame(records)
    for split in ("train", "val", "test"):
        df[df.split == split][["id", "text", "label"]].to_csv(OUT / f"tickets_{split}.csv", index=False)

    print(f"Wrote {n} tickets -> {OUT / 'tickets_all.jsonl'}")
    print("split sizes:", dict(Counter(df.split)))
    print("noisy labels injected (train only):", noisy)


if __name__ == "__main__":
    main()
