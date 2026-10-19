"""Shared paths and hashing for ModelGate."""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    return os.path.join(ROOT, *parts)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


DATA_CSV = p("data", "credit.csv")
TEST_FROZEN = p("data", "test_frozen.csv")
TEST_HASH_FILE = p("records", "test_frozen.sha256")
MODEL_DIR = p("models", "model_v1")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
MODEL_CARD = os.path.join(MODEL_DIR, "model_card.md")
EVAL_RECORD = p("records", "eval_record.json")
DATA_CARD = p("records", "data_card.md")
APPROVALS_LOG = p("records", "approvals.log")
CONFIG = p("gate_config.json")
TARGET = "default_next_month"
