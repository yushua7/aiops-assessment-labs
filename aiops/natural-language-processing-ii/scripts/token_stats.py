#!/usr/bin/env python3
"""Inspect WordPiece behaviour on ops vocabulary and choose a max_length."""
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL = "google/bert_uncased_L-4_H-256_A-4"     # BERT-Mini: 4 layers, 256 hidden, ~11M params
CANDIDATES = [32, 48, 64, 96, 128]


def recommend_max_length(lengths, coverage=0.99):
    for c in CANDIDATES:
        if (lengths <= c).mean() >= coverage:
            return c
    return CANDIDATES[-1]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
    print("== Subword view of operational vocabulary ==")
    for s in ["ECONNREFUSED", "payments-api", "10.42.7.19", "k8s-node-07.prod.local",
              "PG-53300", "CrashLoopBackOff", "disk full"]:
        pieces = tok.tokenize(s)
        print(f"{s:24s} -> {len(pieces):2d} pieces: {pieces}")

    train = pd.read_csv("data/tickets_train.csv")
    enc = tok(list(train.text))["input_ids"]
    lengths = np.array([len(ids) for ids in enc])
    words = train.text.str.split().str.len().to_numpy()

    rec = recommend_max_length(lengths)
    stats = {
        "model": MODEL,
        "min": int(lengths.min()),
        "p50": int(np.percentile(lengths, 50)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "tokens_per_word": round(float(lengths.sum() / words.sum()), 2),
        "unk_tokens": int(sum(ids.count(tok.unk_token_id) for ids in enc)),
        "recommended_max_length": int(rec),
        "truncated_fraction": round(float((lengths > rec).mean()), 4),
    }
    print("\n== Sequence length statistics (train) ==")
    print(json.dumps(stats, indent=2))

    print("\n== Padding / truncation / attention mask demo (max_length=16) ==")
    demo = tok(["disk full", train.text.iloc[0]], padding="max_length",
               truncation=True, max_length=16)
    for ids, mask in zip(demo["input_ids"], demo["attention_mask"]):
        print(tok.convert_ids_to_tokens(ids))
        print(mask)

    with open("reports/token_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\nsaved reports/token_stats.json")


if __name__ == "__main__":
    main()
