# shellcheck shell=bash
# Detection engine: normalise raw events, aggregate per minute, flag anomalies.

normalize_logs() {
  sed -E 's/\r$//; s/[[:space:]]+$//' \
    | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z ' \
    | LC_ALL=C sort -s -k1,1 \
    || true
}

aggregate_minutes() {
  awk '
  {
    minute = substr($1, 1, 16)
    if (!(minute in total)) order[++n] = minute
    total[minute]++
    status = 0; lat = 0
    for (i = 2; i <= NF; i++) {
      split($i, kv, "=")
      if (kv[1] == "status") status = kv[2] + 0
      else if (kv[1] == "latency_ms") lat = kv[2] + 0
    }
    if (status >= 500) err[minute]++
    latsum[minute] += lat
  }
  END {
    for (k = 1; k <= n; k++) {
      m = order[k]
      printf "%s\t%d\t%d\t%.1f\n", m, err[m] + 0, total[m], latsum[m] / total[m]
    }
  }'
}

detect_anomalies() {
  awk -F'\t' \
    -v zt="${Z_THRESHOLD:-3}" -v minerr="${MIN_ERRORS:-3}" -v latms="${LATENCY_MS:-500}" '
  { m[NR] = $1; e[NR] = $2; t[NR] = $3; l[NR] = $4; sum += $2; sq += $2 * $2 }
  END {
    if (NR == 0) exit 0
    mean = sum / NR
    var = sq / NR - mean * mean
    if (var < 0) var = 0
    sd = sqrt(var)
    for (i = 1; i <= NR; i++) {
      z = (sd > 0) ? (e[i] - mean) / sd : 0
      reason = ""
      if (z >= zt && e[i] >= minerr) reason = "error_spike"
      if (l[i] >= latms) reason = (reason == "") ? "high_latency" : reason ",high_latency"
      if (reason != "") printf "%s\t%d\t%d\t%.1f\t%.2f\t%s\n", m[i], e[i], t[i], l[i], z, reason
    }
  }'
}

anomalies_to_json() {
  jq -R -s '
    split("\n") | map(select(length > 0) | split("\t") | {
      minute: .[0],
      errors: (.[1] | tonumber),
      total: (.[2] | tonumber),
      avg_latency_ms: (.[3] | tonumber),
      zscore: (.[4] | tonumber),
      reasons: (.[5] | split(","))
    })'
}
