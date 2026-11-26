#!/usr/bin/env python3
"""Compare lexical (TF-IDF) and semantic (MiniLM) retrieval over the incident KB."""
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def recall_at_k(sim, gold, k):
    topk = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([gold[i] in topk[i] for i in range(len(gold))]))


def main():
    kb = json.load(open("data/kb.json"))
    queries = json.load(open("data/kb_queries.json"))
    ids = [k["id"] for k in kb]
    kb_texts = [k["title"] for k in kb]
    q_texts = [q["text"] for q in queries]
    gold = np.array([ids.index(q["expected"]) for q in queries])

    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(kb_texts)
    sim_lex = (tfidf.transform(q_texts) @ tfidf.transform(kb_texts).T).toarray()

    model = SentenceTransformer(ST_MODEL, device="cpu")
    model.save("models/minilm")                      # local copy for offline serving
    kb_emb = model.encode(kb_texts, normalize_embeddings=True)
    q_emb = model.encode(q_texts, normalize_embeddings=True)
    print("embedding shape:", kb_emb.shape)
    sim_sem = q_emb @ kb_emb.T                       # cosine similarity (unit vectors)

    print(f"\n{'query':<62} {'tfidf':<8} {'minilm':<8} score")
    for i, q in enumerate(q_texts):
        lex, sem = ids[sim_lex[i].argmax()], ids[sim_sem[i].argmax()]
        mark = lambda p: p + ("  " if p == ids[gold[i]] else "x ")
        print(f"{q[:60]:<62} {mark(lex):<8} {mark(sem):<8} {sim_sem[i].max():.2f}")

    metrics = {}
    for name, sim in (("tfidf", sim_lex), ("minilm", sim_sem)):
        metrics[name] = {f"recall@{k}": round(recall_at_k(sim, gold, k), 3) for k in (1, 3)}
    print("\n", json.dumps(metrics))
    with open("reports/search_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
