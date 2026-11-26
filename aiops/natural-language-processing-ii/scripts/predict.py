#!/usr/bin/env python3
import sys

from transformers import pipeline

clf = pipeline("text-classification", model="models/ticket-clf", device=-1)
texts = sys.argv[1:] or [
    "Replication lag on db-03 is 900 seconds for ledger",
    "WAF blocked SQL injection attempt against search-api from 10.1.2.3, rule WAF-942100",
    "Disk usage on stor-11 reached 97 percent on /data",
]
for t, p in zip(texts, clf(texts, truncation=True)):
    print(f"{p['label']:<12} {p['score']:.3f}  {t}")
