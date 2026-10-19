"""Write records/data_card.md from the actual dataset on disk."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_CARD, DATA_CSV, TEST_FROZEN, sha256_file


def main():
    import pandas as pd

    df = pd.read_csv(DATA_CSV)
    test = pd.read_csv(TEST_FROZEN)
    pos = float(df["default_next_month"].mean())
    with open(DATA_CARD, "w") as f:
        f.write(f"""# Data card: credit-risk teaching dataset

- Source: Default of Credit Card Clients, UCI Machine Learning
  Repository (dataset 350). Public, collected 2005, Taiwan card
  clients. Downloaded {date.today().isoformat()} from
  archive.ics.uci.edu.
- File: data/credit.csv, sha256 {sha256_file(DATA_CSV)[:16]}...
- Rows: {len(df)}. Columns: {len(df.columns)} (23 features + target).
- Target: default_next_month, 1 = the client defaulted the next
  month. Positive rate in source: {pos:.3f}.
- Frozen test split: data/test_frozen.csv, {len(test)} rows,
  sha256 {sha256_file(TEST_FROZEN)[:16]}... Split seed 19, stratified,
  frozen at train time; the gate re-checks this hash.
- Features: credit limit, sex, education, marital status, age, six
  months of repayment status, bill amounts and payment amounts.
- Known limits: one country, one era, one bank's book. Class balance
  is skewed (about 22 percent positive). The data carries no names or
  account numbers, but it is still treated as sensitive-shaped:
  public data only, never real client data, in every build here.
- Licence and terms: UCI repository terms; cite Yeh and Lien (2009).
""")
    print(f"data card written: {DATA_CARD}")


if __name__ == "__main__":
    main()
