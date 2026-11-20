# Analysis

1. Highest fail count: **secure-configuration** (89 failures). Layer contribution: {'host': 76, 'container': 2, 'kubernetes': 11}.
2. Unmapped rows: 88 before; after adding the pattern `\bntp\b|chrony|aide|umask|core.?dump|motd` to secure-configuration: 82.
3. A theme with any pass and any fail is "Partial" because a control objective is only met when every in-scope check passes; a mix means some assets or settings still deviate, so claiming "Compliant" would overstate assurance.
   Auditors prefer per-control evidence over a single score (such as kubescape's compliance percentage) because a score averages away critical gaps, hides which control and resource failed, and cannot be traced back to a named requirement or re-verified.
