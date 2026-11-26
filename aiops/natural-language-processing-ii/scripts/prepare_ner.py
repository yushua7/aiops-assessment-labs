#!/usr/bin/env python3
"""Convert char-offset annotations to spaCy DocBin files per split."""
import json
from collections import Counter

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

nlp = spacy.blank("en")
bins = {s: DocBin() for s in ("train", "val", "test")}
skipped, counts = 0, Counter()
with open("data/tickets_all.jsonl") as fh:
    for line in fh:
        rec = json.loads(line)
        doc = nlp.make_doc(rec["text"])
        spans = [doc.char_span(s, e, label=lab) for s, e, lab in rec["entities"]]
        if any(sp is None for sp in spans):     # span does not align with token boundaries
            skipped += 1
            continue
        doc.ents = filter_spans(spans)
        bins[rec["split"]].add(doc)
        counts[rec["split"]] += 1
for split, db in bins.items():
    db.to_disk(f"data/ner_{split}.spacy")
print("docs per split:", dict(counts), "| skipped (misaligned):", skipped)
