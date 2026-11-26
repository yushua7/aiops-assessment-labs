#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/aiops-lab/mlflow"
mkdir -p "$BASE/artifacts"
if curl -sf http://127.0.0.1:5000/health >/dev/null; then
  echo "MLflow already running"; exit 0
fi
source "$HOME/aiops-lab/.venv/bin/activate"
nohup mlflow server \
  --backend-store-uri "sqlite:///$BASE/mlflow.db" \
  --artifacts-destination "$BASE/artifacts" \
  --host 127.0.0.1 --port 5000 --workers 1 \
  > "$BASE/server.log" 2>&1 &
echo "Waiting for MLflow..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:5000/health >/dev/null; then
    echo "MLflow is up on http://127.0.0.1:5000"; exit 0
  fi
  sleep 2
done
echo "MLflow did not start; see $BASE/server.log"; exit 1
