#!/usr/bin/env python3
"""Cluster unlabelled tickets with MiniLM embeddings, with and without masking."""
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, homogeneity_score, silhouette_score

IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
HOST_RE = re.compile(r"\b[a-z0-9]+(?:-[a-z0-9]+)*-\d{2}(?:\.[a-z0-9.]+)?\b")
NUM_RE = re.compile(r"\b\d+\b")
PREFIX_RE = re.compile(r"^(?:urgent:|\[p\d\]|customer reports:|monitoring alert -)\s*")


def normalise(text):
    t = text.lower()
    t = PREFIX_RE.sub("", t, count=1)
    t = IP_RE.sub("<ip>", t)
    t = HOST_RE.sub("<host>", t)
    t = NUM_RE.sub("<num>", t)
    return re.sub(r"\s+", " ", t).strip()


def top_terms(texts, clusters, n=6):
    """Class-based TF-IDF: treat each cluster as one document."""
    ids = sorted(set(clusters))
    docs = [" ".join(t for t, c in zip(texts, clusters) if c == k) for k in ids]
    stop = list(ENGLISH_STOP_WORDS | {"ip", "host", "num", "please", "investigate",
                                      "started", "deploy", "happening", "intermittently",
                                      "raised", "call"})
    vec = TfidfVectorizer(stop_words=stop, token_pattern=r"(?u)\b[a-z][a-z]+\b")
    X = vec.fit_transform(docs).toarray()
    names = vec.get_feature_names_out()
    return {int(k): list(names[np.argsort(X[i])[::-1][:n]]) for i, k in enumerate(ids)}


def evaluate(name, emb, labels, k_final=5):
    print(f"\n== {name} ==")
    sil = {}
    for k in range(3, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
        sil[k] = round(float(silhouette_score(emb, km.labels_, metric="cosine")), 3)
    print("silhouette by k:", sil)
    km = KMeans(n_clusters=k_final, n_init=10, random_state=42).fit(emb)
    res = {"best_k_by_silhouette": max(sil, key=sil.get),
           "silhouette_k5": sil[k_final],
           "ari": round(float(adjusted_rand_score(labels, km.labels_)), 3),
           "homogeneity": round(float(homogeneity_score(labels, km.labels_)), 3)}
    print(res)
    return res, km.labels_


def main():
    assert normalise("[P1] Disk usage on db-07.prod.local reached 93 percent on /data") == \
        "disk usage on <host> reached <num> percent on /data", "normalise() does not match spec"
    df = pd.concat([pd.read_csv("data/tickets_val.csv"), pd.read_csv("data/tickets_test.csv")])
    model = SentenceTransformer("models/minilm", device="cpu")
    raw_texts = df.text.tolist()
    norm_texts = [normalise(t) for t in raw_texts]
    print("example:", raw_texts[0], "\n     -> ", norm_texts[0])

    raw_res, _ = evaluate("raw text", model.encode(raw_texts, normalize_embeddings=True), df.label)
    norm_res, clusters = evaluate("normalised text", model.encode(norm_texts, normalize_embeddings=True), df.label)

    print("\n== cluster topics (c-TF-IDF) vs dominant true label ==")
    df["cluster"] = clusters
    terms = top_terms(norm_texts, clusters)
    for k, words in terms.items():
        sub = df[df.cluster == k]
        dom = sub.label.value_counts()
        print(f"cluster {k} (n={len(sub):3d}, {dom.index[0]} {dom.iloc[0] / len(sub):.0%}): {', '.join(words)}")

    df[["id", "label", "cluster", "text"]].to_csv("reports/clusters.csv", index=False)
    with open("reports/cluster_metrics.json", "w") as fh:
        json.dump({"raw": raw_res, "normalised": norm_res, "topics": terms}, fh, indent=2)


if __name__ == "__main__":
    main()
