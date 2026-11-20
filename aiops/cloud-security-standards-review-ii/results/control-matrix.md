# Acme Payments - Multi-Standard Control Matrix

Generated: 2026-11-20T05:36:37.052354+00:00  
Findings analysed: 532 from 9 evidence files

| Theme | CIS | NIST 800-53 r5 | CSA CCM v4 | Pass | Fail | Fails by layer | Status |
|---|---|---|---|---|---|---|---|
| Secrets and credential management | Kubernetes 5.4.1, 5.1.6; AWS 1.x (credentials) | IA-5, SC-12 | IAM-14, CEK-12 | 14 | 27 | host:23, kubernetes:4 | Partial |
| Least privilege and privileged access | Ubuntu 5.2.x (SSH root login); Docker 5.4/5.5; Kubernetes 5.1.x, 5.2.x; AWS 1.16 | AC-2, AC-6, AC-6(9) | IAM-05, IAM-09, IAM-16 | 47 | 11 | container:1, host:3, kubernetes:7 | Partial |
| Encryption at rest and in transit | AWS 2.1.1, 2.2.1, 2.3.1; Kubernetes 1.2.x TLS flags | SC-8, SC-12, SC-28 | CEK-03, DSP-17 | 28 | 1 | kubernetes:1 | Partial |
| Network segmentation and exposure | AWS 5.2; Docker 2.2; Kubernetes 5.3.x; Ubuntu 3.x | SC-7, AC-4 | IVS-03 | 28 | 9 | container:1, host:4, iac:1, kubernetes:3 | Partial |
| Logging, audit and monitoring | AWS 3.x; Kubernetes 1.2.16-1.2.19; Ubuntu 4.x; Docker 2.x logging | AU-2, AU-6, AU-12 | LOG-03, LOG-08 | 22 | 42 | container:13, host:16, iac:8, kubernetes:5 | Partial |
| Backup and resilience | Docker 2.x live-restore; AWS 2.1.x versioning | CP-9, CP-10 | BCR-08 | 19 | 0 | - | Compliant |
| Vulnerability and patch management | Docker 4.x; Ubuntu 1.9 (updates) | RA-5, SI-2 | TVM-07, TVM-03 | 4 | 6 | container:1, host:5 | Partial |
| Secure baseline configuration | Ubuntu 1.x-6.x; Docker 2.x/3.x/5.x; Kubernetes 1.x/4.x | CM-6, CM-7, CM-8 | IVS-04, CCC-06, CCC-07 | 102 | 89 | container:2, host:76, kubernetes:11 | Partial |
| Unmapped |  |  |  | 52 | 31 | container:4, host:22, iac:1, kubernetes:4 | Partial |

## Open high/critical findings (max 25)

| Source | Layer | ID | Title | Theme |
|---|---|---|---|---|
| openscap | host | gnome_gdm_disable_xdmcp | Disable XDMCP in GDM | unmapped |
| openscap | host | grub2_uefi_password | Set the UEFI Boot Loader Password | secrets-management |
| openscap | host | package_timesyncd_installed | Install the systemd_timesyncd Service | secure-configuration |
| openscap | host | sshd_disable_empty_passwords | Disable SSH Access via Empty Passwords | secrets-management |
| lynis | host | PKGS-7392 | Found one or more vulnerable packages. | vulnerability-mgmt |
| kubescape | kubernetes | C-0012 | Applications credentials in configuration files | secrets-management |
| kubescape | kubernetes | C-0038 | Host PID/IPC privileges | privileged-access |
| kubescape | kubernetes | C-0041 | HostNetwork access | network-exposure |
| kubescape | kubernetes | C-0046 | Insecure capabilities | privileged-access |
| kubescape | kubernetes | C-0057 | Privileged container | privileged-access |
| kubescape | kubernetes | C-0270 | Ensure CPU limits are set | secure-configuration |
| kubescape | kubernetes | C-0271 | Ensure memory limits are set | secure-configuration |
| conftest | kubernetes | org-policy | [CIS K8s 5.2.12][NIST AC-6][CCM IVS-04] Deployment/legacy-web mounts hostPath /etc | secure-configuration |
| conftest | kubernetes | org-policy | [CIS K8s 5.2.2][NIST AC-6][CCM IVS-04] Deployment/legacy-web container web is privileged | privileged-access |
| conftest | kubernetes | org-policy | [CIS K8s 5.4.1][NIST IA-5][CCM IAM-14] Deployment/legacy-web has literal secret in env DB_PASSWORD | secrets-management |
| conftest | kubernetes | org-policy | [NIST SC-6][CCM IVS-04] Deployment/legacy-web container web has no memory limit | secure-configuration |
| conftest | kubernetes | org-policy | [CIS K8s 5.1.1][NIST AC-6][CCM IAM-05] cluster-admin bound to ServiceAccount legacy/default | secrets-management |
