# Advanced Bash Automation for AIOps: A Log-Anomaly Detection and Alerting Script

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
An operations team needs a lightweight anomaly detector that runs on any Linux host without a heavy agent. It has to read events from log files, the systemd journal, and Docker container logs, find abnormal minutes, send alerts reliably, and be tested, linted, and scheduled like production code.

## What I did
1. **Strict-mode CLI:** worked through the pitfalls of `set -Eeuo pipefail` (a failed pipe, unset variables, and `((i++))` killing a script), then linted a defective script with ShellCheck and fixed it. The CLI uses `getopts`, a config file with defaults, config file, and command-line precedence, and documented exit codes (0, 1, 2, 3, 64).
2. **Test data:** explored a deterministic log with an injected incident using `grep`, `sed`, `awk`, and `jq`.
3. **Detection engine:** implemented a per-minute z-score rule in `awk` (population standard deviation, with a guard for a flat baseline) alongside a static latency threshold, and rendered the results as JSON with `jq`.
4. **Multi-source collection:** collected events from a file, the systemd journal (`journalctl -o json`), and a live Docker container, and tested it with an injected fault whose cleanup runs from an `EXIT` trap.
5. **Reliable alerting:** wrote webhook delivery with retry and exponential backoff, idempotency keys for de-duplication, `flock` to stop overlapping runs, a spool file for outages, and `logger` for a local audit trail. I simulated a webhook outage and watched alerts spool and then resend.
6. **Testing:** wrote a bats-core test suite covering unit tests on the functions and integration tests on the CLI, then broke the code on purpose to confirm the tests caught it.
7. **Scheduling:** ran the detector from a systemd service and timer (using `SuccessExitStatus=2`, since exit 2 means "anomalies found") and compared it with cron.

## Results
| Check | Result |
|-------|--------|
| Lab validation script | 16 passed, 0 failed |
| bats test suite | 11 tests, all passing |
| ShellCheck (`-x -S warning`) | Clean on every script |

The detector flags exactly the three injected incident minutes (10:30 to 10:32), each with a z-score of 4.36 and both the error spike and high latency reasons. Raising the z threshold to 5 leaves only the latency rule, which shows why the two rules work together: z-scores are weak on short windows or flat baselines.

The output is saved in [`results/`](results).

## Key concepts
| Area | What it demonstrates |
|------|----------------------|
| Strict mode | Catches many errors, but needs traps and early validation as well |
| Exit codes | 0 ok, 1 error, 2 anomalies found, 3 missing dependency, 64 usage error |
| awk statistics | Associative arrays and an `END` block are enough for z-score detection |
| Alert reliability | Retry with backoff, de-duplication, locking, spooling |
| Testing | bats-core with deterministic fixtures, plus ShellCheck |
| Scheduling | systemd timers add dependency ordering, catch-up runs, and journal logs |

## What I learned
- `((i++))` returns false when `i` is 0, so under `set -e` it stops the script. Use `i=$((i + 1))`.
- A failed `grep` match inside a pipeline can abort a script when `pipefail` is on.
- A health check or a retry loop needs an upper bound. Without the attempt limit, an outage would make the retry loop run forever.
- Keys are recorded only after a successful delivery, which gives at-least-once delivery: an alert may be sent twice, but never lost.
- Exit code 2 is a normal outcome for this tool, so the systemd service has to declare it a success.

## Repo contents
- `lib/detect.sh`: normalisation, per-minute aggregation, z-score detection, and JSON output
- `lib/alert.sh`: webhook delivery with retry and backoff, de-duplication, locking, and spooling
- `tests/detect.bats`: unit and integration tests
- `results/`: validation, bats, and ShellCheck output

These three files are the exercises I completed, and they include scaffolding provided by the course. The command-line wrapper, log generator, collectors, and webhook receiver are not included.

## Tools
Bash, awk, sed, grep, jq, ShellCheck, bats-core, Docker, systemd, cron, curl, flock
