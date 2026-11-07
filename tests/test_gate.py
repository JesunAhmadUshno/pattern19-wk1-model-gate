"""Behaviour tests for the promotion gate (src/gate.py).

Each test builds a throwaway copy of the repo layout in a temp folder,
with a tiny stand-in frozen test file, runs the real gate as a
subprocess, and checks its exit code and output. Standard library only.
Run: python3 -m unittest discover -s tests -v
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROVALS_HEADER = (
    "# ModelGate approvals log. One line per approval, format:\n"
    "# APPROVED <full name> <YYYY-MM-DD>\n"
    "# The gate looks for one APPROVED line. Comments and blanks are "
    "ignored.\n")
FROZEN = b"ID,x,default_next_month\n1,0.5,0\n2,0.7,1\n"
FROZEN_HASH = hashlib.sha256(FROZEN).hexdigest()


def run_gate(root):
    r = subprocess.run([sys.executable, os.path.join(root, "src", "gate.py")],
                       capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


class Sandbox:
    """A minimal repo copy where every evidence record is present and good."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="modelgate-")
        for d in ("src", "data", "records"):
            os.makedirs(os.path.join(self.root, d))
        for f in ("gate.py", "common.py"):
            shutil.copy(os.path.join(REPO, "src", f),
                        os.path.join(self.root, "src", f))
        self.config = {"model_version": "model v1", "floor_auc": 0.65}
        self.eval = {"test_rows": 2, "test_sha256": FROZEN_HASH,
                     "auc": 0.724, "accuracy": 0.811, "floor_auc": 0.65}
        self.write("data/test_frozen.csv", FROZEN)
        self.write("records/test_frozen.sha256", FROZEN_HASH + "\n")
        self.write("records/data_card.md", "# Data card\n")
        self.write("records/approvals.log",
                   APPROVALS_HEADER + "APPROVED Jesun Ahmad Ushno 2026-10-19\n")

    def write(self, rel, content):
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(os.path.join(self.root, rel), mode) as f:
            f.write(content)

    def remove(self, rel):
        os.remove(os.path.join(self.root, rel))

    def run(self):
        self.write("gate_config.json", json.dumps(self.config))
        if self.eval is not None:
            self.write("records/eval_record.json", json.dumps(self.eval))
        return run_gate(self.root)

    def close(self):
        shutil.rmtree(self.root)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.box = Sandbox()

    def tearDown(self):
        self.box.close()

    def assertBlocked(self, reason):
        code, out, err = self.box.run()
        self.assertEqual(code, 1, out + err)
        self.assertIn("PROMOTE model v1: BLOCKED", out, err)
        self.assertIn(reason, out)
        self.assertEqual(err, "", "the gate must block, not crash")
        return out

    def test_complete_records_pass(self):
        code, out, _ = self.box.run()
        self.assertEqual(code, 0, out)
        self.assertTrue(out.endswith("PROMOTE model v1: PASS\n"))

    def test_missing_eval_record_blocks(self):
        self.box.eval = None
        self.assertBlocked("no eval record")

    def test_missing_data_card_blocks(self):
        self.box.remove("records/data_card.md")
        self.assertBlocked("no data card")

    def test_empty_data_card_blocks(self):
        self.box.write("records/data_card.md", "")
        self.assertBlocked("no data card")

    def test_no_approval_line_blocks(self):
        self.box.write("records/approvals.log", APPROVALS_HEADER)
        self.assertBlocked("no approval line")

    def test_malformed_approval_line_blocks(self):
        self.box.write("records/approvals.log",
                       APPROVALS_HEADER + "APPROVED Jesun 19-10-2026\n")
        self.assertBlocked("no approval line")

    def test_tampered_test_file_blocks(self):
        self.box.write("data/test_frozen.csv", FROZEN.replace(b"0.7", b"0.8"))
        self.assertBlocked("eval earned on a swapped test set")

    def test_eval_hash_not_frozen_hash_blocks(self):
        self.box.eval["test_sha256"] = "0" * 64
        self.assertBlocked("eval earned on a swapped test set")

    def test_auc_below_floor_blocks(self):
        self.box.eval["auc"] = 0.6
        out = self.assertBlocked("score below the floor set before training")
        self.assertIn("FAIL (AUC 0.6 below floor 0.65)", out)

    def test_eval_record_cannot_lower_the_floor(self):
        # The floor lives in gate_config.json. An eval record that carries
        # its own, lower floor must not move it.
        self.box.eval["auc"] = 0.6
        self.box.eval["floor_auc"] = 0.5
        self.assertBlocked("score below the floor set before training")

    def test_raised_config_floor_blocks(self):
        self.box.config["floor_auc"] = 0.75
        self.assertBlocked("score below the floor set before training")

    def test_eval_record_without_auc_blocks_instead_of_crashing(self):
        del self.box.eval["auc"]
        self.assertBlocked("eval record has no numeric AUC")

    def test_eval_record_with_text_auc_blocks(self):
        self.box.eval["auc"] = "high"
        self.assertBlocked("eval record has no numeric AUC")

    # json.load accepts NaN and Infinity. NaN < floor is False, so a gate
    # that only checks "below the floor" would let a NaN score through.
    def test_nan_auc_blocks(self):
        self.box.eval["auc"] = float("nan")
        self.assertBlocked("eval record has no numeric AUC")

    def test_infinite_auc_blocks(self):
        self.box.eval["auc"] = float("inf")
        self.assertBlocked("eval record has no numeric AUC")

    def test_auc_above_one_blocks(self):
        self.box.eval["auc"] = 1.5
        self.assertBlocked("eval record has no numeric AUC")

    def test_nan_floor_blocks(self):
        self.box.config["floor_auc"] = float("nan")
        self.assertBlocked("no floor in gate_config.json")

    def test_config_without_floor_blocks_instead_of_crashing(self):
        del self.box.config["floor_auc"]
        self.assertBlocked("no floor in gate_config.json")

    def test_every_record_missing_reports_all_three_reasons(self):
        self.box.eval = None
        self.box.remove("records/data_card.md")
        self.box.write("records/approvals.log", APPROVALS_HEADER)
        out = self.assertBlocked(
            "reasons: no eval record; no data card; no approval line")
        with open(os.path.join(REPO, "demo", "run1_blocked.txt")) as f:
            self.assertEqual(out, f.read())


class ShippedRecordsTest(unittest.TestCase):
    def test_shipped_records_reproduce_the_recorded_pass(self):
        code, out, err = run_gate(REPO)
        self.assertEqual(code, 0, out + err)
        with open(os.path.join(REPO, "demo", "run2_pass.txt")) as f:
            self.assertEqual(out, f.read())


if __name__ == "__main__":
    unittest.main()
