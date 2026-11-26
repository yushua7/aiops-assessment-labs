#!/usr/bin/env python3
"""Fine-tune BERT-Mini for 5-way incident ticket classification (CPU)."""
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

MODEL = "google/bert_uncased_L-4_H-256_A-4"
OUT_DIR = "models/ticket-clf"
set_seed(42)
torch.set_num_threads(2)

with open("reports/token_stats.json") as fh:
    MAX_LEN = int(json.load(fh)["recommended_max_length"])

train_df = pd.read_csv("data/tickets_train.csv")
val_df = pd.read_csv("data/tickets_val.csv")
test_df = pd.read_csv("data/tickets_test.csv")
LABELS = sorted(train_df.label.unique())
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# model_max_length is set explicitly: this checkpoint ships no tokenizer_config,
# so without it truncation=True at serving time would have no upper bound.
tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)


class TicketDataset(Dataset):
    def __init__(self, df):
        self.enc = tokenizer(list(df.text), truncation=True,
                             max_length=MAX_LEN, padding="max_length")
        self.labels = [label2id[l] for l in df.label]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def main():
    print(f"labels={LABELS} max_length={MAX_LEN}")
    train_ds, val_ds, test_ds = TicketDataset(train_df), TicketDataset(val_df), TicketDataset(test_df)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)

    args = TrainingArguments(
        output_dir="work/clf-checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)

    pred = trainer.predict(test_ds)
    y_pred = pred.predictions.argmax(-1)
    y_true = np.array(test_ds.labels)
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    print("TEST accuracy={:.4f} f1_macro={:.4f}".format(
        pred.metrics["test_accuracy"], pred.metrics["test_f1_macro"]))

    print("\n== error analysis: misclassified test tickets ==")
    wrong = np.where(y_pred != y_true)[0]
    for i in wrong[:10]:
        print(f"true={LABELS[y_true[i]]:<11} pred={LABELS[y_pred[i]]:<11} {test_df.text.iloc[i]}")
    print(f"total misclassified: {len(wrong)}")

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    with open("reports/transformer_metrics.json", "w") as fh:
        json.dump({"model": MODEL, "max_length": MAX_LEN, "train_seconds": train_seconds,
                   "accuracy": round(pred.metrics["test_accuracy"], 4),
                   "f1_macro": round(pred.metrics["test_f1_macro"], 4)}, fh, indent=2)
    print(f"saved model to {OUT_DIR}")


if __name__ == "__main__":
    main()
