# shellcheck shell=bash disable=SC2154
# Alert delivery: webhook with retry/backoff, syslog via logger, de-duplication, spooling.

: "${WEBHOOK_URL:=http://127.0.0.1:8080/alert}"
: "${ALERT_RETRIES:=3}"
: "${STATE_DIR:=${LAB_ROOT:-.}/state}"
: "${SOURCE:=file}"

alert_key() {
  printf '%s|%s|%s' "$1" "$2" "$3" | sha256sum | cut -c1-16
}

already_alerted() {
  local key="$1" db="$STATE_DIR/sent.keys"
  [[ -f "$db" ]] && grep -qxF -- "$key" "$db"
}

mark_alerted() {
  printf '%s\n' "$1" >> "$STATE_DIR/sent.keys"
}

build_payload() {
  local minute="$1" errors="$2" total="$3" lat="$4" z="$5" reasons="$6"
  jq -cn \
    --arg host "$(hostname)" --arg source "$SOURCE" --arg minute "$minute" \
    --argjson errors "$errors" --argjson total "$total" \
    --argjson lat "$lat" --argjson z "$z" --arg reasons "$reasons" \
    '{host: $host, source: $source, minute: $minute,
      errors: $errors, total: $total, avg_latency_ms: $lat, zscore: $z,
      reasons: ($reasons | split(",")),
      severity: (if ($reasons | test("error_spike")) then "critical" else "warning" end),
      detected_at: (now | floor | todate)}'
}

# post_with_retry PAYLOAD -> 0 on HTTP 2xx, 1 after ALERT_RETRIES failed attempts.
post_with_retry() {
  local payload="$1" attempt=1 delay=1 code
  while :; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 \
      -H 'Content-Type: application/json' --data "$payload" "$WEBHOOK_URL" 2>/dev/null)" || true
    if [[ "$code" == 2?? ]]; then
      return 0
    fi
    if (( attempt >= ALERT_RETRIES )); then
      log ERROR "webhook delivery failed after $attempt attempt(s) (last HTTP ${code:-000})"
      return 1
    fi
    log WARN "webhook attempt $attempt/$ALERT_RETRIES failed (HTTP ${code:-000}); retrying in ${delay}s"
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
}

send_alerts() {
  local file="$1" minute errors total lat z reasons key payload failed=0 sent=0
  mkdir -p -- "$STATE_DIR"
  exec 9>"$STATE_DIR/alert.lock"
  if ! flock -n 9; then
    log WARN "another instance is sending alerts; skipping this run"
    exec 9>&-
    return 0
  fi
  while IFS=$'\t' read -r minute errors total lat z reasons; do
    [[ -n "$minute" ]] || continue
    key="$(alert_key "$SOURCE" "$minute" "$reasons")"
    if already_alerted "$key"; then
      log INFO "duplicate suppressed: $minute ($reasons)"
      continue
    fi
    payload="$(build_payload "$minute" "$errors" "$total" "$lat" "$z" "$reasons")"
    logger -t aiops-detect -p user.warning -- \
      "anomaly source=$SOURCE minute=$minute errors=$errors/$total avg_latency_ms=$lat z=$z reasons=$reasons"
    if post_with_retry "$payload"; then
      mark_alerted "$key"
      sent=$((sent + 1))
      log INFO "alert sent: $minute ($reasons)"
    else
      printf '%s\n' "$payload" >> "$STATE_DIR/spool.jsonl"
      failed=$((failed + 1))
      log WARN "alert spooled to $STATE_DIR/spool.jsonl: $minute"
    fi
  done < "$file"
  exec 9>&-
  log INFO "alerts sent=$sent spooled=$failed"
  (( failed == 0 ))
}
