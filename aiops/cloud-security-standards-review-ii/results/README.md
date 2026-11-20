# Multi-Standard Cloud Security Review: Benchmarking Host, Container, Kubernetes and IaC Configurations

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi (Cloud Security Standards Review II).

## Scenario
A payments company is moving to a cloud-native stack and must show auditors evidence of a configuration review. The review covers four layers, each against its own benchmark, and ends with one control matrix that maps every finding to CIS, NIST SP 800-53 Rev. 5, and the CSA Cloud Controls Matrix.

| Layer | Benchmark | Tools |
|-------|-----------|-------|
| Host OS | CIS Ubuntu 22.04 Level 1 Server | OpenSCAP, Lynis |
| Containers | CIS Docker Benchmark | docker-bench-security, Trivy |
| Kubernetes | CIS Kubernetes, NSA/CISA guide, Pod Security Standards | kind, kube-bench, kubescape, Conftest |
| Infrastructure as Code | CIS AWS Foundations | Terraform (plan only), Checkov, Trivy, KICS, Conftest/OPA |
| Governance | NIST 800-53, CSA CCM | Python control-mapping script, git |

Each layer followed the same pattern: baseline scan, targeted fix, rescan to prove it, and a recorded exception where a fix wasn't appropriate.

## Results

| Layer | Before | After |
|-------|--------|-------|
| Host (OpenSCAP failed rules) | 110 | 100 |
| Host (Lynis hardening index) | 59 | 60 |
| Docker (docker-bench warnings) | 33 | 22 |
| IaC (Checkov failed checks) | 36 | 5 |
| IaC (Trivy misconfigurations) | 26 | 2 |
| IaC (KICS high-severity) | 12 | 0 |

- **Container images:** Trivy found 349 high or critical vulnerabilities in the legacy `nginx:1.21` image and 68 in the unprivileged Alpine-based image the hardened container uses.
- **Kubernetes:** kube-bench on the local kind cluster showed 60 passes and 11 failures (mostly audit logging and profiling flags, which are kind development defaults). The live-cluster kubescape (NSA framework) score was about 67%. The deliberately insecure legacy manifest scored 45% and the hardened payments manifest scored 100%.
- **Admission control:** the `payments` namespace enforces the `restricted` Pod Security Standard. A privileged test pod was denied with a message listing every violated rule.
- **Policy as code:** Conftest unit tests passed 6 of 6. The baseline Terraform plan failed 11 organisation policies, the legacy Kubernetes manifest failed 5, and the remediated plan passed all of them.
- **Report:** the final script consolidated 532 findings from 9 evidence files into 8 control themes, plus an "unmapped" bucket.
- **Validation:** the validation script passed all 34 checks.

## Problems I ran into
- **Ubuntu 24.04 instead of 22.04.** The lab is written for Ubuntu 22.04, but the lab machine ran 24.04. The `libopenscap8` package no longer exists there, though `oscap` installed from newer packages. The 22.04 benchmark content also reported every rule as "not applicable", so the host was presented as 22.04 only while the scan ran, using a temporary bind mount over `/etc/lsb-release`. This means the host results are indicative, because the benchmark was written for a different Ubuntu release.
- **Trivy 0.50.1 download returned 404.** The release file was no longer available, so I extracted the same version of the binary from its official container image.
- **Two failing validation checks.**
  - KICS still reported two high-severity issues: a KMS key with full permissions and an S3 bucket without MFA delete. I added a scoped key policy and enabled MFA delete in the Terraform. The second fix satisfies the scanner, but enabling MFA delete has extra requirements in real AWS.
  - The Checkov suppression justifications were stored in a different JSON field than the audit check expected, so I normalised the field.

## Deviations I documented
Some benchmark items were deliberately not applied, and each was recorded as an accepted risk with a justification rather than ignored:
- IP forwarding stays on, because Docker and Kubernetes need it on a container host.
- `no-new-privileges` is set per container instead of daemon-wide, because a daemon-wide setting breaks kind's privileged nodes.
- Three Checkov findings were suppressed with written justifications and risk IDs.

## What I learned
- A CIS benchmark gives a technical setting, NIST and CCM give the control objective, and a review links the two so each scanner finding becomes evidence for a named control.
- No single tool covers everything. Three IaC scanners had overlapping and unique findings.
- Fix-then-rescan evidence is more credible than a single scan.
- Admission control and policy checks on `terraform plan` stop bad changes before deployment, and scans confirm the result afterwards.
- A scanner passing doesn't always mean the setting works in the real service.
- The control matrix, not the raw scanner output, is what an auditor reads.

## Repo contents
- `results/control-matrix.md`: findings mapped to CIS, NIST 800-53 and CCM themes
- `results/summary.json` and `results/findings.csv`: the consolidated findings
- `results/analysis.md`: my written analysis
- `results/risk-register.md`: accepted risks with justifications

Course-provided scripts, Terraform, and Rego are not included.

## Tools
OpenSCAP, Lynis, Docker, docker-bench-security, Trivy, kind, kube-bench, kubescape, Terraform, Checkov, KICS, Conftest/OPA, Python, git
