# Docker Deep Dive II: Building, Hardening and Operating a Containerised AIOps Anomaly-Detection Service

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
A data-science team handed over a small anomaly-detection API (FastAPI and a scikit-learn IsolationForest) with a Dockerfile that works but is poor: the image is huge, it runs as root, it bakes junk data and a secret into the image, and every code change rebuilds everything. The job was to turn it into a production-style stack and operate it entirely from the terminal.

## What I did
1. **Profiled the naive image:** measured its size, layers, and history, and found the problems.
2. **Rebuilt it with a multi-stage BuildKit Dockerfile:** with cache mounts, a build secret, OCI labels, a non-root user, and a healthcheck.
3. **Hardened the runtime:** read-only root filesystem, dropped capabilities, no-new-privileges, and memory, CPU, and process limits.
4. **Designed the networks:** a front-end network and an internal back-end network for the Redis cache, with DNS service discovery, and proved the isolation.
5. **Persistence:** wrote data to a named volume, backed it up with a throwaway container, and restored it after a simulated disaster.
6. **Composed the full stack:** API, Redis, and Prometheus, with healthcheck-gated startup, log rotation, and an optional load-generator profile.
7. **Supply chain:** pushed images to a local registry, pulled by immutable digest, scanned with Trivy, produced an SBOM, and ran a CI-style security gate.
8. **Operations:** looked at logs, resource use, and disk use, then built a monitor that feeds live container metrics to the anomaly API and caught an injected CPU incident.

## Results

### Image: naive versus optimised
| | Naive | Multi-stage |
|--|-------|-------------|
| Size | 2.18 GB | 637 MB (about 70% smaller) |
| User | root (uid 0) | 10001:10001 |
| Command | shell form | exec form |
| Secret in image | Yes (token readable with `cat`) | No (not in history, not in `/run/secrets`) |
| Build context | 52.45 MB | 142 bytes |
| Rebuild after a code change | re-ran every step | 1.5 seconds (cached layers) |

The model was trained during the build using a secret mount, and the model's stored fingerprint matched the expected hash of the token, which proves the secret was used without being stored.

### Hardening, proved from inside the container
- A write to `/srv` failed with "Read-only file system", while `/tmp` stayed writable (a tmpfs mount).
- The container showed no capabilities (`CapEff: 0000000000000000`) and `NoNewPrivs: 1`.
- Limits applied: 512 MiB memory, 1 CPU, 128 processes, restart policy `unless-stopped`.
- The process limit worked: a fork test stopped at 123 child processes with "Resource temporarily unavailable", and the next `docker exec` was refused for the same reason while the limit was full.

### Networking
- The back-end network is internal: a container on it resolved the cache by its alias but could not reach the Internet.
- A container on the front-end network could reach the API but could not resolve or reach Redis.
- The API joined both networks, so it reached Redis while Redis stayed hidden from the front end.
- The cache worked: the first request returned `cached: false` and the identical second request returned `cached: true`.

### Storage
A key written to Redis survived deleting and recreating the container, and survived deleting the whole volume after a backup and restore from a tar archive. A container started without `-v` created an anonymous volume, which was removed with `docker rm -v`. On the host the volume files showed an owner named `dnsmasq`, because Redis runs as uid 999 inside the container and the host maps that number to an unrelated name. The number is what matters, not the name.

### Compose stack
All three services reported healthy, started in order (Redis, then the API, then Prometheus), and Prometheus reported both scrape targets as up. With the load-generator profile enabled, the API served 27 normal and 27 anomalous predictions in about 30 seconds. The API ran with a read-only filesystem, all capabilities dropped, a 512 MiB limit, 128 processes, and rotated logs (1 MB, 3 files).

### Supply chain
| | Result |
|--|--------|
| Local registry | tags 1.0 and 1.1 pushed and listed |
| Digest | local digest equal to the registry's digest; the image was pulled and run by digest |
| Trivy, fixable HIGH and CRITICAL | 0 in the Debian 12 base, 7 HIGH in Python packages |
| Trivy, all findings | 2 CRITICAL, 106 HIGH, 158 MEDIUM, 119 LOW, 10 UNKNOWN |
| SBOM | CycloneDX, 164 components (including fastapi 0.110.0, numpy 1.26.4, scikit-learn 1.4.2) |
| CI gate (fixable CRITICAL) | exit code 0 (pipeline proceeds) |

The gate passed because it only looks at CRITICAL issues that have a fix available. The full report still lists unfixed CRITICAL and many HIGH findings, so passing the gate doesn't mean the image has no vulnerabilities, and a real team would decide how to handle those.

### Operations and the AIOps loop
Resource snapshot of the running stack: the API used 94 MiB of its 512 MiB limit (18%), Redis 4.7 MiB, and Prometheus 34.7 MiB.

The monitor sends container CPU, memory, and request latency to the anomaly API:

| Phase | CPU | Monitor result | Exit code |
|-------|-----|----------------|-----------|
| Baseline | about 0.2% | normal | 0 |
| Injected CPU burner | about 101 to 104% | anomaly (all 3 samples) | 2 |
| After restarting the API | 0.2 to 16% | normal | 0 |

The CPU stayed near 100% (not 200%) because of the 1-CPU limit, which contained the runaway process.

### Validation
The lab's validation script passed all 25 checks.

## A problem I ran into
The first run stopped at the Compose step with `can't set distinct values on 'pids_limit' and 'deploy.resources.limits.pids'`. The newer Compose version rejected the file because the process limit was set in two places. I fixed it by setting the CPU, memory, swap, and process limits as top-level service fields instead of under `deploy`, validated the file, and resumed from the Compose step. The rest of the lab then passed.

## What I learned
- Image size and build speed come from layer order and the build context: copy dependency files before source code, and use `.dockerignore`.
- A build secret keeps credentials out of every layer and the image history, unlike `COPY`.
- Hardening should be layered, with limits you can prove from inside the container.
- A tag can move but a digest can't, so deploy by digest.
- A passing security gate depends on how the gate is defined, so read the full report too.
- Feeding container metrics into an anomaly detector and acting on its exit code is a simple, working AIOps loop.

## Repo contents
- `results/`: the Trivy vulnerability report, the CycloneDX SBOM, the gate output, and the recorded image digest

Course-provided source files, the Dockerfile, and the Compose file are not included.

## Tools
Docker, BuildKit, Docker Compose, Redis, Prometheus, Trivy, a local OCI registry, FastAPI, scikit-learn, jq, Bash
