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
