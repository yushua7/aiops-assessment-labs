#!/usr/bin/env bats
# Unit and integration tests for aiops-detect.

bats_require_minimum_version 1.5.0

setup() {
  LAB_ROOT="$(cd "$BATS_TEST_DIRNAME/.." && pwd)"
  source "$LAB_ROOT/lib/common.sh"
  source "$LAB_ROOT/lib/detect.sh"
  source "$LAB_ROOT/lib/alert.sh"
  FIXTURE="$BATS_TEST_TMPDIR/app.log"
  "$LAB_ROOT/bin/gen-logs" > "$FIXTURE"
}

@test "normalize_logs keeps 1200 valid events and drops malformed lines" {
  normalize_logs < "$FIXTURE" > "$BATS_TEST_TMPDIR/events.log"
  [ "$(wc -l < "$BATS_TEST_TMPDIR/events.log")" -eq 1200 ]
  [ "$(grep -c '^###' "$BATS_TEST_TMPDIR/events.log")" -eq 0 ]
}

@test "aggregate_minutes produces 60 rows with correct incident values" {
  normalize_logs < "$FIXTURE" | aggregate_minutes > "$BATS_TEST_TMPDIR/m.tsv"
  [ "$(wc -l < "$BATS_TEST_TMPDIR/m.tsv")" -eq 60 ]
  run awk -F'\t' '$1 == "2024-05-01T10:30" {print $2, $3, $4}' "$BATS_TEST_TMPDIR/m.tsv"
  [ "$output" = "10 20 973.5" ]
}

@test "detect_anomalies flags exactly the three incident minutes" {
  normalize_logs < "$FIXTURE" | aggregate_minutes | detect_anomalies > "$BATS_TEST_TMPDIR/a.tsv"
  [ "$(wc -l < "$BATS_TEST_TMPDIR/a.tsv")" -eq 3 ]
  run cut -f1,5,6 "$BATS_TEST_TMPDIR/a.tsv"
  [ "${lines[0]}" = $'2024-05-01T10:30\t4.36\terror_spike,high_latency' ]
  [ "${lines[2]}" = $'2024-05-01T10:32\t4.36\terror_spike,high_latency' ]
}

@test "detect_anomalies tolerates a flat baseline (sd = 0)" {
  printf '2024-01-01T00:00\t2\t10\t50.0\n2024-01-01T00:01\t2\t10\t50.0\n' > "$BATS_TEST_TMPDIR/flat.tsv"
  run detect_anomalies < "$BATS_TEST_TMPDIR/flat.tsv"
  [ "$status" -eq 0 ]
  [ -z "$output" ]
}

@test "a higher z threshold leaves only the latency rule" {
  normalize_logs < "$FIXTURE" | aggregate_minutes > "$BATS_TEST_TMPDIR/m.tsv"
  Z_THRESHOLD=5 detect_anomalies < "$BATS_TEST_TMPDIR/m.tsv" > "$BATS_TEST_TMPDIR/a.tsv"
  [ "$(cut -f6 "$BATS_TEST_TMPDIR/a.tsv" | sort -u)" = "high_latency" ]
}

@test "alert_key is deterministic and distinguishes minutes" {
  a="$(alert_key file 2024-05-01T10:30 error_spike)"
  b="$(alert_key file 2024-05-01T10:30 error_spike)"
  c="$(alert_key file 2024-05-01T10:31 error_spike)"
  [ "$a" = "$b" ]
  [ "$a" != "$c" ]
  [ "${#a}" -eq 16 ]
}

@test "post_with_retry gives up after ALERT_RETRIES attempts" {
  WEBHOOK_URL="http://127.0.0.1:9/alert" ALERT_RETRIES=2 run post_with_retry '{}'
  [ "$status" -eq 1 ]
  [[ "$output" == *"failed after 2 attempt(s)"* ]]
}

@test "CLI rejects unknown options with exit 64" {
  run "$LAB_ROOT/bin/aiops-detect" -q
  [ "$status" -eq 64 ]
}

@test "CLI dry run exits 2 and prints a JSON array of 3 anomalies" {
  run --separate-stderr "$LAB_ROOT/bin/aiops-detect" -s file -f "$FIXTURE" -n
  [ "$status" -eq 2 ]
  [ "$(jq length <<<"$output")" -eq 3 ]
}

@test "CLI fails with exit 1 for an unreadable file" {
  run "$LAB_ROOT/bin/aiops-detect" -s file -f /nonexistent.log -n
  [ "$status" -eq 1 ]
  [[ "$output" == *"cannot read log file"* ]]
}

@test "CLI tsv output has a header and three data rows" {
  run --separate-stderr "$LAB_ROOT/bin/aiops-detect" -s file -f "$FIXTURE" -o tsv -n
  [ "$status" -eq 2 ]
  [ "${#lines[@]}" -eq 4 ]
  [ "${lines[0]}" = $'minute\terrors\ttotal\tavg_latency_ms\tzscore\treasons' ]
}
