#!/usr/bin/env bash
# Natural Language Processing II - Transformer-Based Incident Intelligence (BERT-Mini, MiniLM, spaCy NER, FastAPI, Docker) - full automation (no set -e on purpose)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export DEBIAN_FRONTEND=noninteractive
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_DOWNLOAD_TIMEOUT=120

log(){ echo; echo "=== $* ==="; }
retry(){ local n=$1 i; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "retry $i/$n: $*"; sleep 5; done; return 1; }
wuntil(){ local t=$1; shift; local end=$((SECONDS+t)); until eval "$*" >/dev/null 2>&1; do [ $SECONDS -ge $end ] && return 1; sleep 3; done; return 0; }

log "Setup: packages, docker, swap"
sudo chmod 666 /var/run/docker.sock 2>/dev/null
retry 5 sudo apt-get -o DPkg::Lock::Timeout=300 update -qq
retry 3 sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq python3-venv python3-pip python3-dev build-essential git curl jq ca-certificates nano
command -v docker >/dev/null || retry 3 sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq docker.io
sudo systemctl enable --now docker >/dev/null 2>&1
sudo usermod -aG docker "$USER" 2>/dev/null
sudo chmod 666 /var/run/docker.sock
if [ "$(swapon --show --noheadings | wc -l)" -eq 0 ]; then
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
fi
free -h; nproc; df -h / | tail -1
python3 --version; docker --version
ss -ltn | grep -qE ":8000 " && echo "WARNING: port 8000 is already in use"
PYV=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "host python = $PYV (Docker base image will match)"

log "Project layout, venv, CPU torch, pinned stack"
mkdir -p ~/nlp2-lab/{data,scripts,models,reports,service,work}
cd ~/nlp2-lab
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip==24.0 "setuptools<70" wheel
retry 3 pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
cat <<'EOF' > requirements.txt
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
joblib==1.4.2
transformers==4.40.2
tokenizers==0.19.1
huggingface-hub==0.23.4
accelerate==0.30.1
sentence-transformers==2.7.0
spacy==3.7.4
fastapi==0.110.3
uvicorn==0.29.0
EOF
retry 3 pip install -r requirements.txt
cat <<'EOF' > ~/nlp2-lab/env.sh
cd ~/nlp2-lab
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HUB_DOWNLOAD_TIMEOUT=120
EOF
source ~/nlp2-lab/env.sh
python - <<'EOF'
import torch, transformers, sentence_transformers, spacy, sklearn, numpy, fastapi
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("sentence-transformers", sentence_transformers.__version__)
print("spacy", spacy.__version__, "| sklearn", sklearn.__version__, "| numpy", numpy.__version__)
print("fastapi", fastapi.__version__)
EOF

log "Task 1: corpus generation and profiling"
cat <<'EOF' > scripts/generate_data.py
#!/usr/bin/env python3
"""Generate a reproducible synthetic corpus of AIOps incident tickets.

Each record: id, split, label (resolver category), text and character-offset
entity annotations for HOST, IP, SERVICE and ERROR_CODE.
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SEED = 42
N_PER_CLASS = 300
NOISE_RATE = 0.03
OUT = Path("data")
rng = random.Random(SEED)

TEMPLATES = {
    "database": [
        "Deadlock detected on {db} database on {host}, error {code} raised by {svc}",
        "Replication lag on {host} is {num} seconds for {db}",
        "{svc} cannot acquire a connection from the pool, {db} reports {code}",
        "Slow queries on {db} at {host} ({ip}), average latency {num} ms",
        "Database {db} on {host} rejected new connections with {code}",
        "Transaction rollback storm in {db}, {svc} logging {code}",
        "{svc} timing out while talking to {db} on {host}",
    ],
    "network": [
        "Packet loss of {pct} percent between {host} and {ip}",
        "{svc} on {host} fails to reach {ip} with {code}",
        "DNS lookups from {host} timing out, upstream resolver {ip} not answering",
        "BGP session flapping on edge router {host}, peer {ip}",
        "High latency on link to {ip}, {svc} reports {code}",
        "Load balancer health checks failing for {host}, {code} from {ip}",
        "{svc} timing out while talking to {host} ({ip})",
    ],
    "storage": [
        "Disk usage on {host} reached {pct} percent on {path}",
        "{svc} write failed on {host} with {code}, volume {path} is read-only",
        "NFS mount {path} on {host} is unresponsive, error {code}",
        "RAID array degraded on {host}, one disk reporting {code}",
        "Backup to {ip} failed because {path} has no space left, {code}",
        "I/O wait above {pct} percent on {host}, storage latency spiking",
    ],
    "application": [
        "{svc} returning {code} to users after deploy on {host}",
        "{svc} pods restarting in a loop on {host}, exit code {code}",
        "Memory usage of {svc} on {host} keeps growing until it is killed with {code}",
        "Unhandled exception in {svc}, stack trace shows {code}",
        "Response time of {svc} above {num} ms, error rate rising with {code}",
        "Feature flag change broke checkout flow in {svc} on {host}",
        "{svc} timing out for users, {code} seen on {host}",
    ],
    "security": [
        "Multiple failed SSH logins on {host} from {ip}, alert {code}",
        "WAF blocked SQL injection attempt against {svc} from {ip}, rule {code}",
        "Suspicious outbound traffic from {host} to {ip} flagged {code}",
        "User {user} attempted privilege escalation on {host}, {code}",
        "Expired TLS certificate on {svc} detected by scanner, {code}",
        "Malware signature found on {host}, quarantine triggered with {code}",
    ],
}
CODES = {
    "database": ["ORA-00060", "PG-53300", "MYSQL-1205", "PG-40P01", "ORA-01555"],
    "network": ["ETIMEDOUT", "ECONNRESET", "EHOSTUNREACH", "NET-0042"],
    "storage": ["ENOSPC", "EIO", "XFS-117", "NFS-STALE"],
    "application": ["HTTP-500", "HTTP-503", "OOM-137", "NPE-001"],
    "security": ["SEC-4011", "AUTH-401", "WAF-942100", "IDS-2019"],
}
HOST_PREFIX = ["db", "web", "app", "edge", "stor", "auth", "mq", "k8s-node", "cache", "lb"]
HOST_SUFFIX = ["", "", ".prod.local", ".dc1.internal"]
SERVICES = ["payments-api", "auth-service", "inventory-svc", "checkout-web",
            "orders-worker", "search-api", "billing-batch", "notification-svc"]
DATABASES = ["orders", "customers", "ledger", "inventory", "sessions"]
PATHS = ["/var/log", "/data", "/mnt/backups", "/var/lib/docker", "/srv/nfs/shared"]
USERS = ["jsmith", "adeleon", "svc_backup", "mkhan"]
PREFIXES = ["", "", "", "URGENT: ", "[P1] ", "[P2] ", "[P3] ",
            "Customer reports: ", "Monitoring alert - "]
SUFFIXES = ["", "", " Please investigate.", " Started after last deploy.",
            " Happening intermittently.", " Raised by on-call."]

PH = re.compile(r"\{(\w+)\}")
ENT_MAP = {"host": "HOST", "ip": "IP", "svc": "SERVICE", "code": "ERROR_CODE"}


def make_values(label):
    return {
        "host": f"{rng.choice(HOST_PREFIX)}-{rng.randint(1, 40):02d}{rng.choice(HOST_SUFFIX)}",
        "ip": f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        "svc": rng.choice(SERVICES),
        "code": rng.choice(CODES[label]),
        "num": str(rng.randint(30, 2000)),
        "pct": str(rng.randint(20, 99)),
        "db": rng.choice(DATABASES),
        "path": rng.choice(PATHS),
        "user": rng.choice(USERS),
    }


def render(template, values):
    """Fill placeholders and record [start, end, LABEL] for entity placeholders."""
    out, spans, pos = "", [], 0
    for m in PH.finditer(template):
        out += template[pos:m.start()]
        key = m.group(1)
        start = len(out)
        out += values[key]
        if key in ENT_MAP:
            spans.append([start, len(out), ENT_MAP[key]])
        pos = m.end()
    out += template[pos:]
    return out, spans


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    for label, templates in TEMPLATES.items():
        for _ in range(N_PER_CLASS):
            body, spans = render(rng.choice(templates), make_values(label))
            prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
            text = prefix + body + suffix
            spans = [[s + len(prefix), e + len(prefix), lab] for s, e, lab in spans]
            if rng.random() < 0.10:          # lower-casing keeps offsets valid
                text = text.lower()
            records.append({"text": text, "label": label, "entities": spans})

    rng.shuffle(records)
    n = len(records)
    n_train, n_val = n * 70 // 100, n * 15 // 100
    labels = sorted(TEMPLATES)
    noisy = 0
    for i, rec in enumerate(records):
        rec["id"] = f"INC{100000 + i}"
        rec["split"] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
        if rec["split"] == "train" and rng.random() < NOISE_RATE:
            rec["label"] = rng.choice([l for l in labels if l != rec["label"]])
            noisy += 1

    with open(OUT / "tickets_all.jsonl", "w") as fh:
        for rec in records:
            fh.write(json.dumps({k: rec[k] for k in ("id", "split", "label", "text", "entities")}) + "\n")
    df = pd.DataFrame(records)
    for split in ("train", "val", "test"):
        df[df.split == split][["id", "text", "label"]].to_csv(OUT / f"tickets_{split}.csv", index=False)

    print(f"Wrote {n} tickets -> {OUT / 'tickets_all.jsonl'}")
    print("split sizes:", dict(Counter(df.split)))
    print("noisy labels injected (train only):", noisy)


if __name__ == "__main__":
    main()
EOF
python scripts/generate_data.py
cat <<'EOF' > scripts/data_quality.py
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
EOF
python scripts/data_quality.py
python - <<'EOF'
import pandas as pd
df = pd.read_json("data/tickets_all.jsonl", lines=True)
assert len(df) == 1500, len(df)
assert df.split.value_counts().to_dict() == {"train": 1050, "val": 225, "test": 225}
assert df.label.nunique() == 5
for t, spans in zip(df.text, df.entities):
    for s, e, _ in spans:
        assert t[s:e] == t[s:e].strip() and e > s
print("Task 1 OK")
EOF
wc -l data/tickets_train.csv data/tickets_val.csv data/tickets_test.csv

log "Task 2: subword tokenization and sequence budget"
cat <<'EOF' > scripts/token_stats.py
#!/usr/bin/env python3
"""Inspect WordPiece behaviour on ops vocabulary and choose a max_length."""
import json

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

MODEL = "google/bert_uncased_L-4_H-256_A-4"     # BERT-Mini: 4 layers, 256 hidden, ~11M params
CANDIDATES = [32, 48, 64, 96, 128]


def recommend_max_length(lengths, coverage=0.99):
    for c in CANDIDATES:
        if (lengths <= c).mean() >= coverage:
            return c
    return CANDIDATES[-1]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, model_max_length=512)
    print("== Subword view of operational vocabulary ==")
    for s in ["ECONNREFUSED", "payments-api", "10.42.7.19", "k8s-node-07.prod.local",
              "PG-53300", "CrashLoopBackOff", "disk full"]:
        pieces = tok.tokenize(s)
        print(f"{s:24s} -> {len(pieces):2d} pieces: {pieces}")

    train = pd.read_csv("data/tickets_train.csv")
    enc = tok(list(train.text))["input_ids"]
    lengths = np.array([len(ids) for ids in enc])
    words = train.text.str.split().str.len().to_numpy()

    rec = recommend_max_length(lengths)
    stats = {
        "model": MODEL,
        "min": int(lengths.min()),
        "p50": int(np.percentile(lengths, 50)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "tokens_per_word": round(float(lengths.sum() / words.sum()), 2),
        "unk_tokens": int(sum(ids.count(tok.unk_token_id) for ids in enc)),
        "recommended_max_length": int(rec),
        "truncated_fraction": round(float((lengths > rec).mean()), 4),
    }
    print("\n== Sequence length statistics (train) ==")
    print(json.dumps(stats, indent=2))

    print("\n== Padding / truncation / attention mask demo (max_length=16) ==")
    demo = tok(["disk full", train.text.iloc[0]], padding="max_length",
               truncation=True, max_length=16)
    for ids, mask in zip(demo["input_ids"], demo["attention_mask"]):
        print(tok.convert_ids_to_tokens(ids))
        print(mask)

    with open("reports/token_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("\nsaved reports/token_stats.json")


if __name__ == "__main__":
    main()
EOF
retry 3 python scripts/token_stats.py 2>&1 | grep -v -i warning
jq '{p95, p99, max, recommended_max_length, truncated_fraction}' reports/token_stats.json
python -c "import json;d=json.load(open('reports/token_stats.json'));assert d['recommended_max_length'] in (32,48,64,96,128) and d['truncated_fraction']<=0.01;print('Task 2 OK')"

log "Task 3: TF-IDF + logistic regression baseline"
cat <<'EOF' > scripts/baseline.py
#!/usr/bin/env python3
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


def show_top_features(pipe, n=6):
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    for i, cls in enumerate(clf.classes_):
        top = names[np.argsort(clf.coef_[i])[::-1][:n]]
        print(f"{cls}: " + " | ".join(top))


def main():
    train = pd.read_csv("data/tickets_train.csv")
    val = pd.read_csv("data/tickets_val.csv")
    test = pd.read_csv("data/tickets_test.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0)),
    ])
    pipe.fit(train.text, train.label)

    val_pred = pipe.predict(val.text)
    test_pred = pipe.predict(test.text)
    metrics = {
        "model": "tfidf-logreg",
        "val_f1_macro": round(f1_score(val.label, val_pred, average="macro"), 4),
        "accuracy": round(accuracy_score(test.label, test_pred), 4),
        "f1_macro": round(f1_score(test.label, test_pred, average="macro"), 4),
    }
    print(classification_report(test.label, test_pred, digits=3))
    print("== confusion matrix (rows=true, cols=pred) ==")
    print(pd.crosstab(test.label, test_pred, rownames=["true"], colnames=["pred"]))
    print("\n== top features per class ==")
    show_top_features(pipe)

    joblib.dump(pipe, "models/baseline.joblib")
    with open("reports/baseline_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n", metrics)


if __name__ == "__main__":
    main()
EOF
python scripts/baseline.py
jq . reports/baseline_metrics.json
python -c "import json;d=json.load(open('reports/baseline_metrics.json'));assert d['f1_macro']>=0.90;print('Task 3 OK')"
ls -lh models/baseline.joblib

log "Task 4: fine-tune BERT-Mini (a few minutes on CPU)"
cat <<'EOF' > scripts/train_classifier.py
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
EOF
retry 2 bash -c 'python scripts/train_classifier.py > work/train.log 2>&1'
grep -E "^labels=|eval_f1_macro|train_runtime|^TEST|misclassified|^true=|saved model" work/train.log | tail -40
cat <<'EOF' > scripts/predict.py
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
EOF
python scripts/predict.py 2>&1 | grep -v -i warning
cat <<'EOF' > data/ood_tickets.csv
text,label
"the postgres primary is refusing writes and the replica is lagging badly",database
"orders table locked and queries are piling up waiting on locks",database
"users in the branch office cannot reach anything because the vpn tunnel keeps dropping",network
"traceroute shows huge latency at the second hop",network
"the volume holding the logs is completely full",storage
"nfs share keeps going stale and the files are unreadable",storage
"after the release the checkout page throws a 500 error",application
"the java service keeps getting oom killed",application
"someone is hammering the login page with thousands of password guesses",security
"antivirus quarantined a trojan on the build server",security
EOF
cat <<'EOF' > scripts/compare_ood.py
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
EOF
python scripts/compare_ood.py 2>&1 | grep -v -i warning
jq . reports/transformer_metrics.json
ls models/ticket-clf
python -c "import json;d=json.load(open('reports/transformer_metrics.json'));assert d['f1_macro']>=0.85;print('Task 4 OK')"

log "Task 5: semantic search and duplicate detection"
cat <<'EOF' > data/kb.json
[
  {"id": "KB-001", "category": "database", "title": "PostgreSQL primary refused new connections because max_connections was exhausted", "resolution": "Add PgBouncer connection pooling and raise max_connections after capacity review"},
  {"id": "KB-002", "category": "storage", "title": "Disk full on the /var/log volume caused the service to crash", "resolution": "Enable logrotate with compression and extend the log volume"},
  {"id": "KB-003", "category": "security", "title": "TLS certificate expired on the public load balancer and clients saw handshake failures", "resolution": "Renew the certificate and automate renewal with cert-manager"},
  {"id": "KB-004", "category": "application", "title": "Kubernetes pod stuck in CrashLoopBackOff after a bad ConfigMap change", "resolution": "Roll back the ConfigMap and add schema validation to CI"},
  {"id": "KB-005", "category": "network", "title": "DNS resolution failures inside the cluster caused by an overloaded CoreDNS", "resolution": "Scale CoreDNS replicas and enable NodeLocal DNSCache"},
  {"id": "KB-006", "category": "application", "title": "Java service OutOfMemoryError with heap exhausted under peak traffic", "resolution": "Fix the cache leak and right-size the JVM heap"},
  {"id": "KB-007", "category": "security", "title": "Brute force login attempts against SSH from a single external IP", "resolution": "Block the source address and enable fail2ban"},
  {"id": "KB-008", "category": "database", "title": "MySQL replica replication lag exceeded ten minutes", "resolution": "Enable parallel replication workers and move reporting queries off the replica"},
  {"id": "KB-009", "category": "network", "title": "Packet loss between data centres caused by a faulty switch port", "resolution": "Replace the optic and move traffic to the redundant uplink"},
  {"id": "KB-010", "category": "security", "title": "NTP clock drift caused token validation errors", "resolution": "Fix chrony configuration and alert on clock offset"},
  {"id": "KB-011", "category": "application", "title": "RabbitMQ message queue backlog growing because consumers cannot keep up", "resolution": "Scale consumers horizontally and set queue length alerts"},
  {"id": "KB-012", "category": "storage", "title": "Backup job failing because object storage credentials were rotated", "resolution": "Update the backup secret and use short-lived credentials from the vault"}
]
EOF
cat <<'EOF' > data/kb_queries.json
[
  {"text": "postgres will not accept any more clients, it says too many connections", "expected": "KB-001"},
  {"text": "no space left on device, the logging partition is at 100 percent", "expected": "KB-002"},
  {"text": "browsers complain that the site's security certificate is out of date", "expected": "KB-003"},
  {"text": "the container keeps restarting over and over since we changed its settings", "expected": "KB-004"},
  {"text": "apps in k8s cannot look up hostnames and name lookups time out", "expected": "KB-005"},
  {"text": "the jvm ran out of memory during the busiest hour", "expected": "KB-006"},
  {"text": "thousands of failed password attempts on port 22 from one address", "expected": "KB-007"},
  {"text": "the read copy of the database is way behind the master", "expected": "KB-008"},
  {"text": "we are dropping packets on the link between our two sites", "expected": "KB-009"},
  {"text": "server time is off and auth tokens are rejected as not yet valid", "expected": "KB-010"},
  {"text": "the work queue keeps piling up because the workers are too slow", "expected": "KB-011"},
  {"text": "nightly backups broke after the storage access keys were changed", "expected": "KB-012"}
]
EOF
jq length data/kb.json data/kb_queries.json
cat <<'EOF' > scripts/semantic_search.py
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
EOF
retry 3 python scripts/semantic_search.py 2>&1 | grep -v -i warning
cat <<'EOF' > scripts/dedup.py
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
EOF
echo "--- default thresholds ---";  python scripts/dedup.py 2>&1 | grep -v -i warning
echo "--- --dup 0.75 ---";          python scripts/dedup.py --dup 0.75 2>&1 | grep -v -i warning
echo "--- --dup 0.45 ---";          python scripts/dedup.py --dup 0.45 2>&1 | grep -v -i warning
jq . reports/search_metrics.json
ls models/minilm | head
python -c "import json;d=json.load(open('reports/search_metrics.json'));assert d['minilm']['recall@3']>=0.83 and d['minilm']['recall@1']>=d['tfidf']['recall@1'];print('Task 5 OK')"

log "Task 6: clustering with masking and c-TF-IDF"
cat <<'EOF' > scripts/cluster.py
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
EOF
python scripts/cluster.py 2>&1 | grep -v -i warning
jq '{raw: .raw, normalised: .normalised}' reports/cluster_metrics.json
python -c "import json;d=json.load(open('reports/cluster_metrics.json'));assert d['normalised']['ari']>=0.25;print('Task 6 OK')"

log "Task 7: custom NER with spaCy (statistical + rules)"
cat <<'EOF' > scripts/prepare_ner.py
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
EOF
python scripts/prepare_ner.py
python -m spacy init config work/ner.cfg --lang en --pipeline ner --optimize efficiency --force
python -m spacy train work/ner.cfg --output models/ner \
  --paths.train data/ner_train.spacy --paths.dev data/ner_val.spacy \
  --training.max_steps 800 --training.eval_frequency 100 --system.seed 42 > work/spacy_train.log 2>&1
tail -15 work/spacy_train.log
ls models/ner
python -m spacy evaluate models/ner/model-best data/ner_test.spacy --output reports/ner_test_metrics.json 2>&1 | grep -v -i warning
cat <<'EOF' > scripts/build_hybrid_ner.py
#!/usr/bin/env python3
import spacy

nlp = spacy.load("models/ner/model-best")
ruler = nlp.add_pipe("entity_ruler", before="ner")
PATTERNS = [
    {"label": "IP", "pattern": [{"TEXT": {"REGEX": r"^(?:\d{1,3}\.){3}\d{1,3}$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^E(?:CONNREFUSED|CONNRESET|TIMEDOUT|HOSTUNREACH|NOSPC|IO|ACCES|PIPE)$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}-[0-9][0-9A-Z]{1,6}$"}}]},
]
assert PATTERNS, "add your patterns"
ruler.add_patterns(PATTERNS)
nlp.to_disk("models/ner-hybrid")
print("pipeline:", nlp.pipe_names, "| patterns:", len(ruler.patterns))
EOF
python scripts/build_hybrid_ner.py
python -m spacy evaluate models/ner-hybrid data/ner_test.spacy --output reports/ner_hybrid_metrics.json 2>&1 | grep "NER F"
cat <<'EOF' > scripts/entities_demo.py
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
EOF
python scripts/entities_demo.py
python - <<'EOF'
import json, spacy
m = json.load(open("reports/ner_test_metrics.json"))
assert m["ents_f"] >= 0.90, m["ents_f"]
nlp = spacy.load("models/ner-hybrid")
assert nlp.pipe_names == ["entity_ruler", "ner"]
doc = nlp("Firewall dropped traffic from 192.168.200.17 to cache-west-a, code ECONNREFUSED")
got = {(e.text, e.label_) for e in doc.ents}
assert ("192.168.200.17", "IP") in got and ("ECONNREFUSED", "ERROR_CODE") in got, got
print("Task 7 OK, test ents_f =", round(m["ents_f"], 3))
EOF

log "Task 8: FastAPI service, local test, Docker image"
touch service/__init__.py
cat <<'EOF' > service/app.py
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
EOF
docker rm -f ticket-nlp >/dev/null 2>&1
nohup uvicorn service.app:app --host 127.0.0.1 --port 8000 > work/uvicorn.log 2>&1 &
echo $! > work/uvicorn.pid
wuntil 240 "curl -sf http://127.0.0.1:8000/health" || { echo "local API did not start:"; tail -30 work/uvicorn.log; }
curl -s http://127.0.0.1:8000/health | jq
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"URGENT: Database orders on db-07.prod.local rejected new connections with PG-53300, payments-api cannot get a connection from 10.20.30.40"}' | jq
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' -d '{"text":""}'
python -c 'import json;print(json.dumps({"text":"disk full on /var/log " * 200}))' > work/long.json
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d @work/long.json | jq '{category, confidence, latency_ms}'
kill "$(cat work/uvicorn.pid)"; sleep 2

cat <<'EOF' > service/requirements-serve.txt
numpy==1.26.4
scikit-learn==1.4.2
transformers==4.40.2
tokenizers==0.19.1
huggingface-hub==0.23.4
sentence-transformers==2.7.0
spacy==3.7.4
fastapi==0.110.3
uvicorn==0.29.0
EOF
cat <<'EOF' > .dockerignore
.venv
work
reports
scripts
**/__pycache__
models/ner
models/baseline.joblib
EOF
# base image matches the host Python minor version
cat <<'EOF' > Dockerfile
FROM python:PYVER-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    MODEL_DIR=/app/models DATA_DIR=/app/data TORCH_THREADS=2
WORKDIR /app
COPY service/requirements-serve.txt .
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt
COPY service/ service/
COPY models/ticket-clf models/ticket-clf
COPY models/ner-hybrid models/ner-hybrid
COPY models/minilm models/minilm
COPY data/kb.json data/kb.json
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
sed -i "s/PYVER/$PYV/" Dockerfile
head -1 Dockerfile
docker build --provenance=false --sbom=false -t ticket-nlp:1.0 . > work/docker_build.log 2>&1 \
  || docker build -t ticket-nlp:1.0 . > work/docker_build.log 2>&1
tail -3 work/docker_build.log
docker images ticket-nlp
docker run -d --name ticket-nlp -p 8000:8000 --memory 2g ticket-nlp:1.0
wuntil 360 '[ "$(docker inspect -f "{{.State.Health.Status}}" ticket-nlp)" = healthy ]' || { echo "container not healthy:"; docker logs --tail 30 ticket-nlp; }
docker ps --filter name=ticket-nlp --format '{{.Names}} {{.Status}}'
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"postgres is out of connection slots and new sessions are refused","top_k":1}' \
  | jq '{category, duplicate_of, top: .similar_incidents[0].id}'
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{time_total}\n' -X POST http://127.0.0.1:8000/analyze \
    -H 'Content-Type: application/json' -d '{"text":"Disk usage on stor-11 reached 97 percent on /data"}'
done | awk '{s+=$1} END {printf "mean latency: %.1f ms over %d requests\n", s/NR*1000, NR}'
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' ticket-nlp
docker inspect -f '{{.State.Health.Status}}' ticket-nlp
docker exec ticket-nlp python -c "import os;print(os.environ['HF_HUB_OFFLINE'])"
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"Multiple failed SSH logins on auth-02 from 10.9.8.7, alert SEC-4011"}' \
  | jq -e '.category == "security" and ([.entities[].label] | index("IP") != null)' && echo "Task 8 OK"

log "Validation (lab validator)"
cat <<'EOF' > ~/nlp2-lab/validate.sh
#!/usr/bin/env bash
cd ~/nlp2-lab || exit 1
source .venv/bin/activate
pass=0; fail=0
check() {
  local desc="$1"; shift
  if "$@" > /dev/null 2>&1; then echo "[PASS] $desc"; pass=$((pass+1))
  else echo "[FAIL] $desc"; fail=$((fail+1)); fi
}
J='import json,sys; d=json.load(open(sys.argv[1]));'
check "T1 corpus has 1500 tickets"           test "$(wc -l < data/tickets_all.jsonl)" -eq 1500
check "T1 split CSVs exist"                  test -s data/tickets_test.csv
check "T2 token budget chosen"               python -c "$J assert d['recommended_max_length'] in (32,48,64,96,128)" reports/token_stats.json
check "T3 baseline macro-F1 >= 0.90"         python -c "$J assert d['f1_macro']>=0.90" reports/baseline_metrics.json
check "T4 transformer macro-F1 >= 0.85"      python -c "$J assert d['f1_macro']>=0.85" reports/transformer_metrics.json
check "T4 model artefacts saved"             test -f models/ticket-clf/config.json
check "T4 OOD comparison recorded"           test -s reports/ood_comparison.json
check "T5 MiniLM recall@3 >= 0.83"           python -c "$J assert d['minilm']['recall@3']>=0.83" reports/search_metrics.json
check "T5 embedding model saved locally"     test -f models/minilm/modules.json
check "T6 normalised ARI >= 0.25"            python -c "$J assert d['normalised']['ari']>=0.25" reports/cluster_metrics.json
check "T7 NER test F1 >= 0.90"               python -c "$J assert d['ents_f']>=0.90" reports/ner_test_metrics.json
check "T7 hybrid pipeline has entity_ruler"  python -c "import spacy; assert spacy.load('models/ner-hybrid').pipe_names==['entity_ruler','ner']"
check "T8 container healthy"                 bash -c '[ "$(docker inspect -f "{{.State.Health.Status}}" ticket-nlp 2>/dev/null)" = healthy ]'
check "T8 /analyze returns category+entities" bash -c "curl -sf -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{\"text\":\"payments-api on web-12.prod.local returned HTTP-503 to 10.20.30.40\"}' | jq -e '.category and (.entities|length>=3)'"
check "T8 empty input rejected with 422"     bash -c "[ \"\$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' -d '{\"text\":\"\"}')\" = 422 ]"
echo "----------------------------------------"
echo "Passed: $pass  Failed: $fail"
[ "$fail" -eq 0 ]
EOF
chmod +x ~/nlp2-lab/validate.sh
~/nlp2-lab/validate.sh
echo "NLP2 LAB DONE"
# END_OF_NLP2
