# Kubernetes Administrator II: Operating a Multi-Node Cluster for a Payments Platform

Hands-on lab from the AIOps Additional Assessment Labs at Al Nafi.

## Scenario
A fintech start-up is moving its "payments" workloads onto a self-managed Kubernetes cluster. Before go-live, I had to show I could operate the cluster safely: onboard an operator with certificates and least-privilege access, govern resources, place workloads, provide persistent storage, lock down network traffic, perform node maintenance without losing availability, back up and restore the control-plane datastore (etcd), and diagnose failures.

The cluster was built with kind: one control-plane node and two workers (Kubernetes v1.31.0), all running as containers on a single Ubuntu server, with a real kubelet, a kubeadm-built control plane, and real etcd.

## What I did
1. **Cluster and control plane:** built the 3-node cluster, inspected the control-plane static pods and certificate expiry (all control-plane certificates had 364 days left), and created a static pod on a worker. The Kubernetes mirror pod was recreated after I deleted it, and the pod only disappeared when I removed its manifest from the node.
2. **Identity and RBAC:** created a user `dana` through the CertificateSigningRequest API (subject CN `dana`, group `ops-team`, certificate valid for 24 hours), gave the user a namespaced role for Deployments, and gave the group read access to nodes. Created a ServiceAccount `ci-bot` that can only read pods.
3. **Governance and placement:** added a LimitRange and ResourceQuota to the `payments` namespace, labelled nodes, and tainted one worker as dedicated to data.
4. **Storage:** created a StorageClass with the `Retain` reclaim policy and ran a 2-replica StatefulSet with a volume per replica.
5. **Networking:** exposed the services and applied default-deny network policies in both directions, then added only the traffic that is needed.
6. **Maintenance:** protected the API with a PodDisruptionBudget, then cordoned and drained a worker.
7. **Disaster recovery:** took an etcd snapshot, deleted resources, stopped the control plane, restored into a new data directory, and repointed etcd at it.
8. **Break-fix:** installed metrics-server, then diagnosed and fixed five injected faults.

## Results

### Access control
| Identity | Action | Result |
|----------|--------|--------|
| dana (user) | Create Deployments in `payments` | Allowed |
| dana | Scale Deployments | Allowed |
| dana | Delete pods | Denied |
| dana | List secrets | Denied |
| dana (via group `ops-team`) | List nodes | Allowed |
| ci-bot (ServiceAccount) | List pods (direct API call with a token) | Allowed (`PodList`) |
| ci-bot | List secrets | HTTP 403 Forbidden |

Authentication and authorization are separate: right after the certificate was approved, dana could prove who they were (`auth whoami` showed user `dana`, groups `ops-team` and `system:authenticated`) but could do nothing until a Role was bound.

### Governance and placement
The LimitRange gave pods with no resource settings a default request of 100m CPU and 64Mi memory, and a limit of 200m CPU and 128Mi, which I confirmed on a running pod. The three `api` pods landed on `ops-worker`, because the control plane and the data node were both tainted. The `cache` landed on the tainted `ops-worker2` only because it had both a toleration and a node-affinity rule. The toleration allows it onto the node and the affinity attracts it there.

### Storage
Both StatefulSet claims (200Mi each) bound with the `retain-local` class. After I deleted `ledger-0`, its journal still contained the earlier record plus a new "started" line, so data survived the pod. I deleted a scratch claim and saw its volume go to `Released` with its data kept, cleared the claim reference so it became `Available`, and then removed the volume and its data on the node by hand.

### Network policies (six policies in total)
| Test | Expected | Result |
|------|----------|--------|
| frontend to api | allowed | nginx page returned |
| intruder to api | blocked | failed (exit 1) |
| probe (labelled for cache access) to cache | allowed | `+PONG` |
| frontend to cache | blocked | failed (exit 1) |

### Node maintenance
The PodDisruptionBudget (`minAvailable: 2`) initially allowed one disruption. When I drained `ops-worker`, the first evictions were refused with "Cannot evict pod as it would violate the pod's disruption budget" and were retried until replacement pods were running. The drain then completed. The StatefulSet pod `ledger-0` was left `Pending` during the drain, because local volumes tie a pod to its node, and it came back after the node was uncordoned. The ledger journal still showed 3 start records afterwards, and the PDB showed 3 healthy against 2 required.

### etcd backup and restore
- The snapshot was 3.6 MB, with 1,621 keys at revision 1609, and I also copied it off the node.
- After the snapshot I deleted the `api` Deployment and a ConfigMap, and created a new ConfigMap.
- I stopped the control plane by moving the static pod manifests away, restored the snapshot into a new data directory, and edited the etcd manifest to use it.
- After the control plane came back, the deleted Deployment and ConfigMap (`release-info`, version `2024.10-r1`) were back, and the ConfigMap created after the snapshot was gone. A restore rolls back all cluster state to the snapshot point.

### Break-fix
Five faults were injected, and I fixed all five:

| Fault | Symptom | Fix |
|-------|---------|-----|
| Image tag typo | `ImagePullBackOff` | corrected the image tag |
| Service selector did not match pod labels | no endpoints | corrected the selector |
| Memory request larger than any node | `Pending` | set realistic resource requests |
| Missing environment variable | crash loop (`APP_MODE not set`) | set the variable |
| Kubelet config file moved on a node | node problem | restored the file and restarted the kubelet |

At the end all deployments were available, the Service had two endpoints, and a test request returned the nginx page. `kubectl top` showed the control-plane node at 5% CPU and 654 Mi of memory, with the workers much lighter.

### Validation
The lab's validation script passed all 28 checks.

## Things I noticed
- **One interim check failed.** My run script's own check after the maintenance step reported a failure ("only cache on ops-worker2"). The output right before it still shows old pods in `Terminating` state on that node, which is the likely cause, since the final validation later passed all 28 checks.
- **The drain order differed from the lab text.** The lab expects the drain to stall until the data-node taint is removed. In my run the taint was removed first, so the drain only paused briefly on the disruption budget and then finished.
- **The client was newer than the cluster.** The `kubectl` client reported v1.36.2 against a v1.31.0 cluster, which is outside the usual one-version skew support. Everything still worked, but a real team should keep them close.

## What I learned
- The control-plane components are static pods, and editing their manifests is how you reconfigure or stop them on a kubeadm cluster.
- A toleration permits a pod onto a tainted node, and only node affinity pulls it there.
- Default-deny egress blocks DNS too, and policies select the backend pods, not the Service.
- A PodDisruptionBudget protects availability but doesn't create capacity.
- Local volumes pin stateful pods to a node, so they can't move during maintenance.
- An etcd restore is a rollback of everything, so it has to be rehearsed and communicated first.

## Repo contents
- `results/`: cluster state captured at the end of the lab

Course-provided manifests and scripts are not included. Private keys and the etcd snapshot are deliberately not published.

## Tools
Kubernetes, kind, kubectl, etcdctl and etcdutl, containerd, Docker, metrics-server, Bash
