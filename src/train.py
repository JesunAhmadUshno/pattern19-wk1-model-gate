"""Train the credit-risk scorer and freeze the test split.

Writes:
  data/test_frozen.csv          the frozen test rows (never trained on)
  records/test_frozen.sha256    hash of the frozen test file
  models/model_v1/model.pkl     the trained model
  models/model_v1/model_card.md the model card

It deliberately writes NO eval record, NO data card and NO approval:
those are the gate's evidence, produced by their own steps.
"""
import json
import os
import pickle
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONFIG, DATA_CSV, MODEL_CARD, MODEL_DIR, MODEL_FILE,
                    TARGET, TEST_FROZEN, TEST_HASH_FILE, sha256_file)


def main():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(DATA_CSV):
        raise SystemExit("data/credit.csv missing. Run: python src/get_data.py")

    cfg = json.load(open(CONFIG))
    df = pd.read_csv(DATA_CSV)
    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET].astype(int)

    train, test = train_test_split(
        df, test_size=0.20, random_state=19, stratify=y)
    os.makedirs(os.path.dirname(TEST_FROZEN), exist_ok=True)
    test.to_csv(TEST_FROZEN, index=False)
    test_hash = sha256_file(TEST_FROZEN)
    os.makedirs(os.path.dirname(TEST_HASH_FILE), exist_ok=True)
    with open(TEST_HASH_FILE, "w") as f:
        f.write(test_hash + "\n")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=19))
    model.fit(train[features], train[TARGET].astype(int))

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": model, "features": features,
                     "version": cfg["model_version"]}, f)

    data_hash = sha256_file(DATA_CSV)
    with open(MODEL_CARD, "w") as f:
        f.write(f"""# Model card: {cfg['model_version']}

- Algorithm: logistic regression (standardized features), scikit-learn.
- Job: score the risk that a credit card client defaults next month.
- Trained: {date.today().isoformat()} (seed 19, fixed).
- Data: data/credit.csv, sha256 {data_hash[:16]}...
- Train rows: {len(train)}. Frozen test rows: {len(test)}.
- Frozen test file: data/test_frozen.csv, sha256 {test_hash[:16]}...
- Features: {len(features)} columns from the source table (limits,
  demographics, six months of payment history, bill and payment amounts).
- Promotion floor (set before training, gate_config.json): test AUC
  must be at least {cfg['floor_auc']}.
- Limits: a teaching model on one public 2005-era dataset. Not a
  credit decision tool. Not for real applicants, ever.
""")
    print(f"trained {cfg['model_version']} on {len(train)} rows")
    print(f"frozen test: {len(test)} rows, sha256 {test_hash[:16]}...")
    print(f"model card: models/model_v1/model_card.md")


if __name__ == "__main__":
    main()
