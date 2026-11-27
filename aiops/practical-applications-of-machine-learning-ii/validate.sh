#!/usr/bin/env bash
cd "$HOME/aiops-lab" || exit 1
PASS=0; FAIL=0
check() { local d="$1"; shift; if "$@" >/dev/null 2>&1; then echo "[PASS] $d"; PASS=$((PASS+1)); else echo "[FAIL] $d"; FAIL=$((FAIL+1)); fi; }
MLF=http://127.0.0.1:5000/api/2.0/mlflow
alias_version() { curl -s "$MLF/registered-models/alias?name=infra-anomaly-detector&alias=$1" | jq -r .model_version.version; }
runs_count() {
  local exp; exp=$(curl -s "$MLF/experiments/get-by-name?experiment_name=infra-anomaly" | jq -r .experiment.experiment_id)
  curl -s -X POST "$MLF/runs/search" -H 'Content-Type: application/json' \
    -d "{\"experiment_ids\":[\"$exp\"],\"max_results\":200}" | jq '.runs | length'
}
prom() { curl -s -G http://127.0.0.1:9090/api/v1/query --data-urlencode "query=$1" | jq -r '.data.result[0].value[1]'; }

check "T1 metrics.csv has 20160 rows + header"       test "$(wc -l < data/metrics.csv)" -eq 20161
check "T1 28 labelled incidents"                     test "$(jq length data/incidents.json)" -eq 28
check "T1 app.log generated"                         test -s data/app.log
check "T2 baseline sweep saved (5 thresholds)"       test "$(jq length results/baseline.json)" -eq 5
check "T3 MLflow server healthy"                     curl -sf http://127.0.0.1:5000/health
check "T3 at least 6 tracked runs"                   test "$(runs_count)" -ge 6
check "T3 champion alias exists"                     test -n "$(alias_version champion)" -a "$(alias_version champion)" != "null"
check "T4 log templates mined (8)"                   test "$(jq .n_templates results/log_anomaly.json)" -eq 8
check "T4 log anomaly test F1 >= 0.8"                jq -e '.test.f1 >= 0.8' results/log_anomaly.json
check "T5 GBM forecast beats seasonal naive"         jq -e '.gbm.mae < .naive.mae' results/forecast.json
check "T6 service /health returns 200"               curl -sf http://127.0.0.1:8000/health
check "T6 container healthcheck healthy"             test "$(docker inspect -f '{{.State.Health.Status}}' anomaly-svc)" = "healthy"
check "T6 features.py identical in service"          diff -q src/features.py service/features.py
check "T7 Prometheus scraping service"               test "$(prom 'up{job="anomaly-svc"}')" = "1"
check "T7 alert rules loaded"                        test "$(curl -s http://127.0.0.1:9090/api/v1/rules | jq '[.data.groups[].rules[]] | length')" -eq 3
check "T7 drift detected on latency_ms"              jq -e '.metrics[] | select(.metric=="latency_ms") | .status == "significant"' results/drift_report.json
check "T8 promotion decision recorded"               jq -e '.decision' results/promotion.json
check "T8 'previous' alias exists for rollback"      test "$(alias_version previous)" != "null"
check "T8 service runs the current champion"         test "$(curl -s http://127.0.0.1:8000/health | jq -r .model_version)" = "$(alias_version champion)"

echo; echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
