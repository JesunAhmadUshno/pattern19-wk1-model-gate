# Model card: model v1

- Algorithm: logistic regression (standardized features), scikit-learn.
- Job: score the risk that a credit card client defaults next month.
- Trained: 2026-10-19 (seed 19, fixed).
- Data: data/credit.csv, sha256 84ba892a2a55a0d7...
- Train rows: 24000. Frozen test rows: 6000.
- Frozen test file: data/test_frozen.csv, sha256 b76b978228076841...
- Features: 24 columns from the source table (limits,
  demographics, six months of payment history, bill and payment amounts).
- Promotion floor (set before training, gate_config.json): test AUC
  must be at least 0.65.
- Limits: a teaching model on one public 2005-era dataset. Not a
  credit decision tool. Not for real applicants, ever.
