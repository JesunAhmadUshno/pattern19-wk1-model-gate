#!/usr/bin/env bash
# The done test, run in the open.
# Run 1: train only. The gate must print BLOCKED, with the reasons.
# Then the evidence steps run, one approval line is added, and
# Run 2 must print PASS. Both outputs are saved under demo/.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY src/get_data.py || exit 1
$PY src/train.py || exit 1

echo "--- RUN 1: records incomplete ---"
$PY src/gate.py | tee demo/run1_blocked.txt

$PY src/evaluate.py || exit 1
$PY src/data_card.py || exit 1
grep -q "^APPROVED " records/approvals.log || \
  echo "APPROVED Jesun Ahmad Ushno 2026-10-19" >> records/approvals.log

echo "--- RUN 2: records complete ---"
$PY src/gate.py | tee demo/run2_pass.txt
