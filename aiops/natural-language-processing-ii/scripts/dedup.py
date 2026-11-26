#!/usr/bin/env python3
"""Classify incoming tickets as DUPLICATE / RELATED / NEW against the KB."""
import argparse
import json

from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--dup", type=float, default=0.60, help="cosine >= dup -> DUPLICATE")
ap.add_argument("--related", type=float, default=0.40, help="cosine >= related -> RELATED")
a = ap.parse_args()

kb = json.load(open("data/kb.json"))
model = SentenceTransformer("models/minilm", device="cpu")
kb_emb = model.encode([k["title"] for k in kb], normalize_embeddings=True)
incoming = [
    "postgres is out of connection slots and new sessions are refused",
    "the /var/log partition filled up again on web-03",
    "certificate on the public endpoint expired this morning",
    "printer on floor 3 has run out of toner",
    "quarterly budget spreadsheet needs review",
]
for text in incoming:
    sims = kb_emb @ model.encode([text], normalize_embeddings=True)[0]
    best = int(sims.argmax())
    s = float(sims[best])
    verdict = "DUPLICATE" if s >= a.dup else "RELATED" if s >= a.related else "NEW"
    print(f"{verdict:<10} {s:.2f}  {kb[best]['id']}  {text}")
