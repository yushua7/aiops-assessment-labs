#!/usr/bin/env bash
set -u
cd ~/aiops-lab
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PASS=0; FAIL=0
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "[PASS] $desc"; PASS=$((PASS+1));
  else echo "[FAIL] $desc"; FAIL=$((FAIL+1)); fi
}

check "T1 four datasets generated" test -s data/telemetry_v1.csv -a -s data/telemetry_v2.csv -a -s data/telemetry_v1_live.csv -a -s data/telemetry_v2_live.csv
check "T1 manifest records >= 4 versions with hashes" bash -c '[ "$(jq "[.[] | select(.sha256 | length == 64)] | length" data/manifest.json)" -ge 4 ]'
check "T1 data validation passes on training data" python ml/validate_data.py data/telemetry_v1.csv data/telemetry_v2.csv
check "T2 MLflow tracking server healthy" curl -sf http://127.0.0.1:5000/health
check "T2 >= 12 finished experiment runs" python -c "import mlflow,sys; df=mlflow.search_runs(experiment_names=['incident-predictor'], filter_string=\"attributes.status = 'FINISHED'\"); sys.exit(0 if len(df)>=12 else 1)"
check "T3/T8 registry: champion=2, previous=1" python -c "import sys; from mlflow import MlflowClient as C; c=C(); sys.exit(0 if str(c.get_model_version_by_alias('incident-predictor','champion').version)=='2' and str(c.get_model_version_by_alias('incident-predictor','previous').version)=='1' else 1)"
check "T3/T8 exported artifact is model v2" bash -c '[ "$(jq -r .model_version serving/model/metadata.json)" = "2" ] && test -s serving/model/model.joblib'
check "T4 API unit tests pass" bash -c 'cd serving && python -m pytest -q'
check "T5 images 1.0.0 and 1.1.0 exist" docker image inspect incident-api:1.0.0 incident-api:1.1.0
check "T5 image runs as non-root" bash -c '[ "$(docker image inspect incident-api:1.1.0 --format "{{.Config.User}}")" = "10001" ]'
check "T6 kind cluster aiops exists" bash -c 'kind get clusters | grep -qx aiops'
check "T6 readiness probe on /readyz" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].readinessProbe.httpGet.path}")" = "/readyz" ]'
check "T6 liveness probe on /healthz" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].livenessProbe.httpGet.path}")" = "/healthz" ]'
check "T6 memory limit set" bash -c '[ -n "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].resources.limits.memory}")" ]'
check "T7 PSI self-test passes" python ml/drift.py --self-test
check "T7 baseline window shows no significant drift" jq -e '.drift_detected == false' reports/drift_baseline.json
check "T7 current window flags req_per_sec drift" jq -e '.features.req_per_sec.status == "significant"' reports/drift_report.json
check "T8 promotion gate approved" jq -e '.passed == true' reports/promotion_report.json
check "T8 deployment runs incident-api:1.1.0 with 2 ready replicas" bash -c '[ "$(kubectl -n aiops-ml get deploy incident-api -o jsonpath="{.spec.template.spec.containers[0].image}/{.status.readyReplicas}")" = "incident-api:1.1.0/2" ]'
check "T8 pods serve model version 2" bash -c 'kubectl -n aiops-ml exec deploy/incident-api -- python -c "import urllib.request,json;print(json.load(urllib.request.urlopen(\"http://127.0.0.1:8000/model\"))[\"model_version\"])" | grep -qx 2'

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
