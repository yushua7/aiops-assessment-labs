#!/usr/bin/env bash
# NLP II final fix 3: retrain NER (known-good config), merge tok2vec into ner, patch saved config refs, verify reload, rebuild container
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 HF_HUB_DOWNLOAD_TIMEOUT=120
log(){ echo; echo "=== $* ==="; }
wuntil(){ local t=$1; shift; local end=$((SECONDS+t)); until eval "$*" >/dev/null 2>&1; do [ $SECONDS -ge $end ] && return 1; sleep 3; done; return 0; }
die(){ echo "FATAL: $*"; exit 1; }
sudo chmod 666 /var/run/docker.sock 2>/dev/null
cd ~/nlp2-lab && source .venv/bin/activate

log "Cleanup (including stale metrics from earlier runs)"
pkill -f "uvicorn service.app" 2>/dev/null
docker rm -f ticket-nlp >/dev/null 2>&1
rm -rf models/ner models/ner-hybrid reports/ner_test_metrics.json reports/ner_hybrid_metrics.json
free -h | head -2

log "Task 7: train NER with the standard config (about 2 minutes)"
python -m spacy init config work/ner.cfg --lang en --pipeline ner --optimize efficiency --force > /dev/null 2>&1
python -m spacy train work/ner.cfg --output models/ner \
  --paths.train data/ner_train.spacy --paths.dev data/ner_val.spacy \
  --training.max_steps 800 --training.eval_frequency 100 --system.seed 42 > work/spacy_train.log 2>&1
echo "spacy train exit code: $?"
tail -12 work/spacy_train.log
[ -d models/ner/model-best ] || { echo "---- full train log ----"; tail -60 work/spacy_train.log; die "training did not produce models/ner/model-best"; }
python -m spacy evaluate models/ner/model-best data/ner_test.spacy --output reports/ner_test_metrics.json 2>&1 | grep -E "NER (P|R|F)"
[ -s reports/ner_test_metrics.json ] || die "evaluation did not write ner_test_metrics.json"

log "Hybrid pipeline: merge tok2vec into ner, add entity_ruler, patch saved config"
cat <<'EOF' > scripts/build_hybrid_ner.py
#!/usr/bin/env python3
import json
import pathlib
import re

import spacy
from spacy.util import load_config

PATTERNS = [
    {"label": "IP", "pattern": [{"TEXT": {"REGEX": r"^(?:\d{1,3}\.){3}\d{1,3}$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^E(?:CONNREFUSED|CONNRESET|TIMEDOUT|HOSTUNREACH|NOSPC|IO|ACCES|PIPE)$"}}]},
    {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}-[0-9][0-9A-Z]{1,6}$"}}]},
]
assert PATTERNS, "add your patterns"

SRC = "models/ner/model-best"
DST = "models/ner-hybrid"
orig = load_config(f"{SRC}/config.cfg", interpolate=True)   # fully resolved values

nlp = spacy.load(SRC)
if "tok2vec" in nlp.pipe_names:
    nlp.replace_listeners("tok2vec", "ner", ["model.tok2vec"])   # ner now owns a copy of the encoder
    nlp.remove_pipe("tok2vec")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(PATTERNS)
nlp.to_disk(DST)

# replace_listeners leaves ${components.tok2vec....} references in the saved config; resolve them to literal values
cfg_path = pathlib.Path(DST) / "config.cfg"
text = cfg_path.read_text()


def resolve(m):
    node = orig
    for part in m.group(1).split("."):
        node = node[part]
    return json.dumps(node)


patched, n = re.subn(r"\$\{([^}]+)\}", resolve, text)
cfg_path.write_text(patched)
print(f"patched {n} config reference(s)")

check = spacy.load(DST)
print("pipeline:", check.pipe_names, "| patterns:", len(check.get_pipe("entity_ruler").patterns))
EOF
python scripts/build_hybrid_ner.py || die "hybrid build failed"
python -m spacy evaluate models/ner-hybrid data/ner_test.spacy --output reports/ner_hybrid_metrics.json 2>&1 | grep "NER F"
python scripts/entities_demo.py
python - <<'EOF'
import json, spacy
m = json.load(open("reports/ner_test_metrics.json"))
assert m["ents_f"] >= 0.90, m["ents_f"]
nlp = spacy.load("models/ner-hybrid")
assert nlp.pipe_names == ["entity_ruler", "ner"], nlp.pipe_names
doc = nlp("Firewall dropped traffic from 192.168.200.17 to cache-west-a, code ECONNREFUSED")
got = {(e.text, e.label_) for e in doc.ents}
assert ("192.168.200.17", "IP") in got and ("ECONNREFUSED", "ERROR_CODE") in got, got
print("Task 7 OK, test ents_f =", round(m["ents_f"], 3))
EOF
[ -d models/ner-hybrid ] || die "models/ner-hybrid missing"

log "Permissions for the non-root container user"
chmod -R a+rX models/ticket-clf models/ner-hybrid models/minilm data/kb.json service
find models/ticket-clf models/ner-hybrid models/minilm data/kb.json service ! -perm -o=r | head
echo "(no files listed above = all world-readable)"

log "Task 8: local API test"
nohup uvicorn service.app:app --host 127.0.0.1 --port 8000 > work/uvicorn.log 2>&1 &
echo $! > work/uvicorn.pid
wuntil 300 "curl -sf http://127.0.0.1:8000/health" || { echo "local API did not start:"; tail -30 work/uvicorn.log; die "local API failed"; }
curl -s http://127.0.0.1:8000/health | jq -c
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"payments-api on web-12.prod.local returned HTTP-503 to 10.20.30.40"}' | jq -c '{category, entities: [.entities[]|{text,label}]}'
curl -s -o /dev/null -w 'empty-input status=%{http_code}\n' -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' -d '{"text":""}'
kill "$(cat work/uvicorn.pid)" 2>/dev/null; pkill -f "uvicorn service.app" 2>/dev/null; sleep 3

log "Task 8: rebuild image and run container"
docker build --provenance=false --sbom=false -t ticket-nlp:1.0 . > work/docker_build.log 2>&1 \
  || docker build -t ticket-nlp:1.0 . > work/docker_build.log 2>&1
tail -3 work/docker_build.log
docker images ticket-nlp
docker run -d --name ticket-nlp -p 8000:8000 --memory 2g ticket-nlp:1.0
wuntil 420 '[ "$(docker inspect -f "{{.State.Health.Status}}" ticket-nlp)" = healthy ]' || { echo "container not healthy:"; docker logs --tail 40 ticket-nlp; }
docker ps --filter name=ticket-nlp --format '{{.Names}} {{.Status}}'
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"postgres is out of connection slots and new sessions are refused","top_k":1}' \
  | jq -c '{category, duplicate_of, top: .similar_incidents[0].id}'
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{time_total}\n' -X POST http://127.0.0.1:8000/analyze \
    -H 'Content-Type: application/json' -d '{"text":"Disk usage on stor-11 reached 97 percent on /data"}'
done | awk '{s+=$1} END {printf "mean latency: %.1f ms over %d requests\n", s/NR*1000, NR}'
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' ticket-nlp
docker exec ticket-nlp python -c "import os;print('HF_HUB_OFFLINE=',os.environ['HF_HUB_OFFLINE'])"
curl -s -X POST http://127.0.0.1:8000/analyze -H 'Content-Type: application/json' \
  -d '{"text":"Multiple failed SSH logins on auth-02 from 10.9.8.7, alert SEC-4011"}' \
  | jq -e '.category == "security" and ([.entities[].label] | index("IP") != null)' && echo "Task 8 OK"

log "Validation"
~/nlp2-lab/validate.sh | tee work/validate_out.txt
if grep -q "\[FAIL\]" work/validate_out.txt; then
  log "DIAGNOSTICS"
  grep "\[FAIL\]" work/validate_out.txt
  docker ps -a --filter name=ticket-nlp
  docker logs --tail 30 ticket-nlp 2>&1
  tail -10 work/uvicorn.log
else
  echo "ALL CHECKS PASSED - 0 failed"
fi
echo "NLP2 FINAL3 DONE"
# END_OF_FINAL3
