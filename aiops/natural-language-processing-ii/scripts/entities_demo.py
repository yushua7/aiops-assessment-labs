#!/usr/bin/env python3
import spacy

PROBES = [
    "Firewall dropped traffic from 192.168.200.17 to cache-west-a, code ECONNREFUSED",
    "payments-api on web-12.prod.local returned HTTP-503 to 10.20.30.40",
    "Lots of errors today, nobody knows why",
]


def extract_entities(doc):
    out = {}
    for ent in doc.ents:
        out.setdefault(ent.label_, set()).add(ent.text)
    return {lab: sorted(out[lab]) for lab in sorted(out)}


for path in ("models/ner/model-best", "models/ner-hybrid"):
    nlp = spacy.load(path)
    print(f"== {path} ==")
    for text in PROBES:
        print(f"  {text}\n    -> {extract_entities(nlp(text))}")
