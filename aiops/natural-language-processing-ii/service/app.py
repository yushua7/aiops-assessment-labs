"""Ticket intelligence API: classify, extract entities, find similar incidents."""
import json
import os
import time

import numpy as np
import spacy
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import pipeline

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")
DUP_THRESHOLD = float(os.environ.get("DUP_THRESHOLD", "0.60"))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

classifier = pipeline("text-classification", model=os.path.join(MODEL_DIR, "ticket-clf"), device=-1)
nlp = spacy.load(os.path.join(MODEL_DIR, "ner-hybrid"))
embedder = SentenceTransformer(os.path.join(MODEL_DIR, "minilm"), device="cpu")
with open(os.path.join(DATA_DIR, "kb.json")) as fh:
    KB = json.load(fh)
KB_EMB = embedder.encode([k["title"] for k in KB], normalize_embeddings=True)

app = FastAPI(title="Ticket NLP service", version="1.0.0")


class TicketIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)
    top_k: int = Field(3, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "kb_size": len(KB),
            "labels": sorted(classifier.model.config.label2id),
            "ner_pipeline": nlp.pipe_names}


@app.post("/analyze")
def analyze(ticket: TicketIn):
    if not ticket.text.strip():
        raise HTTPException(status_code=422, detail="text is blank")
    t0 = time.perf_counter()
    pred = classifier(ticket.text, truncation=True)[0]
    doc = nlp(ticket.text)
    entities = [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                for e in doc.ents]
    scores = KB_EMB @ embedder.encode([ticket.text], normalize_embeddings=True)[0]
    order = np.argsort(-scores)[: ticket.top_k]
    similar = [{"id": KB[i]["id"], "title": KB[i]["title"],
                "resolution": KB[i]["resolution"], "score": round(float(scores[i]), 3)}
               for i in order]
    return {
        "category": pred["label"],
        "confidence": round(float(pred["score"]), 3),
        "entities": entities,
        "similar_incidents": similar,
        "duplicate_of": similar[0]["id"] if similar[0]["score"] >= DUP_THRESHOLD else None,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
