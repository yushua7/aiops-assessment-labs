#!/usr/bin/env bash
# (Re)start a port-forward to the incident-api Service on localhost:8080.
pkill -f "kubectl -n aiops-ml port-forward" 2>/dev/null || true
sleep 1
nohup kubectl -n aiops-ml port-forward svc/incident-api 8080:80 > /tmp/pf.log 2>&1 &
for _ in $(seq 1 15); do
  if curl -sf http://127.0.0.1:8080/readyz >/dev/null; then
    echo "port-forward ready on http://127.0.0.1:8080"; exit 0
  fi
  sleep 1
done
echo "port-forward failed; see /tmp/pf.log"; exit 1
