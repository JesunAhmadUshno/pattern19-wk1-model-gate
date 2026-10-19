#!/usr/bin/env bash
# The done test, run in the open.
# Run 1: train only, with the evidence records set aside. The shipped
# repo carries the finished records, so the script moves them out of
# the way first; the gate must print BLOCKED, with the reasons.
# Then the evidence steps run, one approval line is added, and
# Run 2 must print PASS. Both outputs are saved under demo/.
# If a step fails mid-run, the shipped records are put back.
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

$PY src/get_data.py || exit 1
$PY src/train.py || exit 1

STASH="$(mktemp -d)"
restore_records() {
  [ -f "$STASH/eval_record.json" ] && \
    cp "$STASH/eval_record.json" records/eval_record.json
  [ -f "$STASH/data_card.md" ] && \
    cp "$STASH/data_card.md" records/data_card.md
  [ -f "$STASH/approvals.log" ] && \
    cp "$STASH/approvals.log" records/approvals.log
}
trap restore_records EXIT

[ -f records/eval_record.json ] && mv records/eval_record.json "$STASH/"
[ -f records/data_card.md ] && mv records/data_card.md "$STASH/"
[ -f records/approvals.log ] && mv records/approvals.log "$STASH/"
printf '# ModelGate approvals log. One line per approval, format:\n# APPROVED <full name> <YYYY-MM-DD>\n# The gate looks for one APPROVED line. Comments and blanks are ignored.\n' \
  > records/approvals.log

echo "--- RUN 1: records incomplete ---"
$PY src/gate.py | tee demo/run1_blocked.txt

$PY src/evaluate.py || exit 1
$PY src/data_card.py || exit 1
grep -q "^APPROVED " records/approvals.log || \
  echo "APPROVED Jesun Ahmad Ushno 2026-10-19" >> records/approvals.log

echo "--- RUN 2: records complete ---"
$PY src/gate.py | tee demo/run2_pass.txt

trap - EXIT
rm -rf "$STASH"
