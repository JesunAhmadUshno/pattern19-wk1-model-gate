"""The promotion gate.

A model is promoted only if all three evidence records check out:
  1. an eval record whose score beats the floor set before training,
     earned on the frozen test set (hash re-verified here);
  2. a data card;
  3. an approval line in records/approvals.log.

The check order is fixed: read records, decide, and only then report.
A gate that cannot say no is a decoration. Exit 0 = PASS, 1 = BLOCKED.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (APPROVALS_LOG, CONFIG, DATA_CARD, EVAL_RECORD,
                    TEST_FROZEN, TEST_HASH_FILE, sha256_file)


def check_eval():
    if not os.path.exists(EVAL_RECORD):
        return False, "MISSING (records/eval_record.json not found)", \
            "no eval record"
    rec = json.load(open(EVAL_RECORD))
    frozen = open(TEST_HASH_FILE).read().strip()
    actual = sha256_file(TEST_FROZEN)
    if rec.get("test_sha256") != frozen or actual != frozen:
        return False, "FAIL (test set hash mismatch: the scored set " \
            "is not the frozen set)", "eval earned on a swapped test set"
    floor = rec.get("floor_auc")
    if rec.get("auc", 0) < floor:
        return False, f"FAIL (AUC {rec['auc']} below floor {floor})", \
            "score below the floor set before training"
    return True, (f"OK (AUC {rec['auc']} >= floor {floor}, "
                  f"{rec['test_rows']} frozen rows, hash verified)"), ""


def check_data_card():
    if not os.path.exists(DATA_CARD) or os.path.getsize(DATA_CARD) == 0:
        return False, "MISSING (records/data_card.md not found)", \
            "no data card"
    return True, "OK (records/data_card.md)", ""


def check_approval():
    if os.path.exists(APPROVALS_LOG):
        for line in open(APPROVALS_LOG):
            m = re.match(r"^APPROVED\s+(.+?)\s+(\d{4}-\d{2}-\d{2})\s*$",
                         line.strip())
            if m:
                return True, f"OK ({m.group(1)}, {m.group(2)})", ""
    return False, "MISSING (no APPROVED line in records/approvals.log)", \
        "no approval line"


def main():
    cfg = json.load(open(CONFIG))
    version = cfg["model_version"]
    checks = [
        ("eval record", check_eval()),
        ("data card", check_data_card()),
        ("approval line", check_approval()),
    ]
    print("ModelGate promotion check")
    print(f"model: {version}")
    reasons = []
    for i, (name, (ok, detail, reason)) in enumerate(checks, 1):
        print(f"[{i}/3] {name} ... {detail}")
        if not ok:
            reasons.append(reason)
    if reasons:
        print(f"PROMOTE {version}: BLOCKED")
        print("reasons: " + "; ".join(reasons))
        return 1
    print(f"PROMOTE {version}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
