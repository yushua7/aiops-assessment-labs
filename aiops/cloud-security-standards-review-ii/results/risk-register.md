# Risk register (CCM CCC-08 exception management)

| ID | Check | Resource | Justification | Compensating control | Owner | Review date |
|---|---|---|---|---|---|---|
| RISK-001 | CIS Docker 2.18 no-new-privileges (daemon-wide) | Docker daemon | Daemon-wide setting breaks kind privileged node containers needed for the Kubernetes review | no-new-privileges:true enforced per container (app-hardened) | platform-team | 2027-05-20 |
| RISK-002 | CIS Ubuntu 3.2.2 IP forwarding disabled | Host sysctl net.ipv4.ip_forward | Host is a container node; Docker and Kubernetes require IPv4 forwarding | Redirect and source-routing sysctls hardened; host firewall and security groups restrict ingress | platform-team | 2027-05-20 |
| RISK-004 | CKV_AWS_144 | aws_s3_bucket.logs | Cross-region replication not required for lab log data | Versioning and KMS encryption enabled | CISO | 2027-05-20 |
| RISK-005 | CKV2_AWS_62 | aws_s3_bucket.logs | Event notifications not required for log bucket | Public access block and access review | CISO | 2027-05-20 |
| RISK-006 | CKV2_AWS_5 | aws_security_group.web | False positive: group is attached to aws_db_instance.payments via vpc_security_group_ids (checkov graph check misses the reference); suppressed in code | Plan-level Conftest policy and KICS confirm attachment and rules | security-lead | 2027-05-20 |
