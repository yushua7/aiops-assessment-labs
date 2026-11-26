#!/usr/bin/env python3
"""Champion/challenger quality gate. Exit 0 = promoted, 2 = rejected."""
import argparse
import datetime as dt
import json
import sys

from mlflow import MlflowClient

from common import MODEL_NAME
from evaluate import evaluate_alias


def gate(champion, challenger, min_gain, max_regression):
    """Decide whether the challenger may replace the champion.

    Returns (passed: bool, reasons: list[str]).
    """
    reasons = []

    gain = challenger["v2_pr_auc"] - champion["v2_pr_auc"]
    ok_gain = gain >= min_gain
    reasons.append(f"{'PASS' if ok_gain else 'FAIL'} recent-data gain {gain:+.4f} >= {min_gain:.4f}")

    regression = champion["v1_pr_auc"] - challenger["v1_pr_auc"]
    ok_regression = challenger["v1_pr_auc"] >= champion["v1_pr_auc"] - max_regression
    reasons.append(f"{'PASS' if ok_regression else 'FAIL'} v1 regression {regression:+.4f} <= {max_regression:.4f}")

    return (ok_gain and ok_regression), reasons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--max-regression", type=float, default=0.05)
    args = p.parse_args()

    client = MlflowClient()
    champion = evaluate_alias("champion")
    challenger = evaluate_alias("challenger")
    print(f"{'metric':<10} {'champion v' + champion['version']:>14} "
          f"{'challenger v' + challenger['version']:>16}")
    for m in ["v1_pr_auc", "v2_pr_auc", "v1_roc_auc", "v2_roc_auc", "v2_f1"]:
        print(f"{m:<10} {champion[m]:>14.4f} {challenger[m]:>16.4f}")

    passed, reasons = gate(champion, challenger, args.min_gain, args.max_regression)
    for r in reasons:
        print(" ", r)

    report = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "champion": champion, "challenger": challenger, "passed": passed,
              "reasons": reasons, "min_gain": args.min_gain,
              "max_regression": args.max_regression}
    with open("reports/promotion_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    if not passed:
        client.set_model_version_tag(MODEL_NAME, challenger["version"], "promotion", "rejected")
        print(f"REJECTED: champion stays at version {champion['version']}")
        sys.exit(2)

    client.set_registered_model_alias(MODEL_NAME, "previous", champion["version"])
    client.set_registered_model_alias(MODEL_NAME, "champion", challenger["version"])
    client.delete_registered_model_alias(MODEL_NAME, "challenger")
    client.set_model_version_tag(MODEL_NAME, challenger["version"], "promotion", "approved")
    print(f"PROMOTED: champion -> version {challenger['version']} "
          f"(previous -> version {champion['version']})")


if __name__ == "__main__":
    main()
