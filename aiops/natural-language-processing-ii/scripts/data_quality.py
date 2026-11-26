#!/usr/bin/env python3
from collections import Counter

import pandas as pd

df = pd.read_json("data/tickets_all.jsonl", lines=True)
print("rows:", len(df))
print("\n== label x split ==")
print(pd.crosstab(df.label, df.split, margins=True))
df["n_words"] = df.text.str.split().str.len()
print("\n== words per ticket ==")
print(df.n_words.describe().round(1).to_string())
print("\nexact duplicate texts:", int(df.text.duplicated().sum()))
ents = Counter(lab for spans in df.entities for _, _, lab in spans)
print("entity mentions:", dict(sorted(ents.items())))
bad = sum(1 for t, spans in zip(df.text, df.entities) for s, e, _ in spans
          if e <= s or t[s:e] != t[s:e].strip())
print("bad spans:", bad)
print("\n== sample ==")
for _, r in df.head(3).iterrows():
    print(f"[{r.label}] {r.text}")
    print("   ", [(r.text[s:e], lab) for s, e, lab in r.entities])
