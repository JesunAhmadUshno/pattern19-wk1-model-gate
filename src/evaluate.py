"""Score the model on the frozen test split and write the eval record.

The eval record carries the frozen test hash, so the gate can prove the
score was earned on the same test set that was frozen at train time.
"""
import json
import os
import pickle
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONFIG, EVAL_RECORD, MODEL_FILE, TARGET, TEST_FROZEN,
                    sha256_file)


def main():
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    cfg = json.load(open(CONFIG))
    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model, features = bundle["model"], bundle["features"]

    test = pd.read_csv(TEST_FROZEN)
    test_hash = sha256_file(TEST_FROZEN)
    proba = model.predict_proba(test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(test[TARGET].astype(int), proba)
    acc = accuracy_score(test[TARGET].astype(int), pred)

    record = {
        "model": cfg["model_version"],
        "evaluated": date.today().isoformat(),
        "test_file": "data/test_frozen.csv",
        "test_rows": int(len(test)),
        "test_sha256": test_hash,
        "auc": round(float(auc), 3),
        "accuracy": round(float(acc), 3),
        "floor_auc": cfg["floor_auc"],
    }
    os.makedirs(os.path.dirname(EVAL_RECORD), exist_ok=True)
    with open(EVAL_RECORD, "w") as f:
        json.dump(record, f, indent=2)
    print(f"eval record written: AUC {record['auc']} on "
          f"{record['test_rows']} frozen rows")


if __name__ == "__main__":
    main()
