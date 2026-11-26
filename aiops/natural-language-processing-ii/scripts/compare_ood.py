#!/usr/bin/env python3
import json

import joblib
import pandas as pd
from transformers import pipeline

ood = pd.read_csv("data/ood_tickets.csv")
base = joblib.load("models/baseline.joblib")
clf = pipeline("text-classification", model="models/ticket-clf", device=-1)
ood["baseline"] = base.predict(ood.text)
ood["transformer"] = [p["label"] for p in clf(list(ood.text), truncation=True)]
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)
print(ood[["label", "baseline", "transformer", "text"]].to_string(index=False))
res = {m: round(float((ood[m] == ood.label).mean()), 2) for m in ("baseline", "transformer")}
print("\nOOD accuracy:", res)
with open("reports/ood_comparison.json", "w") as fh:
    json.dump(res, fh, indent=2)
