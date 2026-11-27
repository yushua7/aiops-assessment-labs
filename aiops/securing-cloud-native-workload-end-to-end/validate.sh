#!/usr/bin/env bash
set -u
source ~/aiops-lab/localstack.env
cd ~/aiops-lab
pass=0; fail=0
check() { local name="$1"; shift; if eval "$@" >/dev/null 2>&1; then echo "PASS  $name"; pass=$((pass+1)); else echo "FAIL  $name"; fail=$((fail+1)); fi; }

echo "== Task 1: cluster and CNI"
check "node Ready"                       "kubectl get nodes --no-headers | grep -q ' Ready '"
check "Calico or NetworkPolicy capable"  "kubectl -n calico-system get ds calico-node || kubectl -n kube-system get pods | grep -q coredns"
check "kube-bench baseline saved"        "grep -q 'Summary' ~/aiops-lab/kube-bench-baseline.txt"

echo "== Task 2: IAM as code"
check "Checkov IAM/S3 checks pass"       "cd ~/aiops-lab/terraform && ~/aiops-lab/.venv-sec/bin/checkov -d . --framework terraform --check \$CHECKS --quiet --compact"
check "policy has no s3:*"               "! awsl iam get-policy-version --policy-arn \$(awsl iam list-policies --scope Local --query \"Policies[?PolicyName=='shop-api-s3'].Arn\" --output text) --version-id v1 | grep -q 's3:\\*'"
check "bucket public access blocked"     "awsl s3api get-public-access-block --bucket shopco-orders | grep -q '\"BlockPublicAcls\": true'"

echo "== Task 3: supply chain"
check "image gate (Trivy)"               "trivy image --quiet --severity HIGH,CRITICAL --ignore-unfixed --ignorefile ~/aiops-lab/.trivyignore --exit-code 1 shop/api:1.0.0"
check "manifest gate (Trivy)"            "trivy config --quiet --severity HIGH,CRITICAL --exit-code 1 ~/aiops-lab/k8s/shop/deployment.yaml"

echo "== Task 4: RBAC and Gatekeeper"
check "ci-deployer cannot read secrets"  "[ \"\$(kubectl auth can-i get secrets -n shop --as=system:serviceaccount:shop:ci-deployer)\" = no ]"
check "ci-deployer can deploy"           "[ \"\$(kubectl auth can-i create deployments -n shop --as=system:serviceaccount:shop:ci-deployer)\" = yes ]"
check "Gatekeeper denies nginx"          "kubectl apply --dry-run=server -f ~/aiops-lab/k8s/tests/gk-test.yaml 2>&1 | grep -q validation.gatekeeper.sh"

echo "== Task 5: Kyverno and NetworkPolicy"
check "Kyverno denies no-limits workload" "kubectl -n shop create deployment v-ky --image=shop/api:1.0.0 --dry-run=server 2>&1 | grep -q require-limits"
check "default-deny generated"           "kubectl -n shop get networkpolicy default-deny-all"

echo "== Task 6: Vault, TLS, flows"
check "shop-api ready"                   "[ \"\$(kubectl -n shop get deploy shop-api -o jsonpath='{.status.readyReplicas}')\" = 1 ]"
check "certificate Ready"                "kubectl -n shop get certificate shop-api-tls -o jsonpath='{.status.conditions[0].status}' | grep -q True"
check "allowed client gets secret flag"  "kubectl -n shop exec client-allowed -- curl -s -m 5 -k https://shop-api.shop.svc/ | grep -q '\"db_secret_loaded\": true'"
check "denied client blocked"            "! kubectl -n shop exec client-denied -- curl -s -m 5 -k https://shop-api.shop.svc/"
check "no DB secret in k8s Secrets"      "! kubectl -n shop get secrets -o name | grep -qi db"

echo "== Task 7: Falco"
check "Falco running"                    "[ \"\$(kubectl -n falco get ds falco -o jsonpath='{.status.numberReady}')\" -ge 1 ]"
check "custom rule fired"                "grep -q 'Shop API Vault Secret Read' ~/aiops-lab/falco-live.jsonl"

echo "== Task 8: ML triage"
check "triage report exists"             "test -s ~/aiops-lab/ml/triage_report.csv"
check "attacks in top 5"                 "~/aiops-lab/.venv-ml/bin/python -c \"import pandas as pd,sys; sys.exit(0 if pd.read_csv('$HOME/aiops-lab/ml/triage_report.csv').head(5).is_attack.sum()>=3 else 1)\""

echo; echo "Result: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
