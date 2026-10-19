# Data card: credit-risk teaching dataset

- Source: Default of Credit Card Clients, UCI Machine Learning
  Repository (dataset 350). Public, collected 2005, Taiwan card
  clients. Downloaded 2026-10-19 from
  archive.ics.uci.edu.
- File: data/credit.csv, sha256 84ba892a2a55a0d7...
- Rows: 30000. Columns: 25 (23 features + target).
- Target: default_next_month, 1 = the client defaulted the next
  month. Positive rate in source: 0.221.
- Frozen test split: data/test_frozen.csv, 6000 rows,
  sha256 b76b978228076841... Split seed 19, stratified,
  frozen at train time; the gate re-checks this hash.
- Features: credit limit, sex, education, marital status, age, six
  months of repayment status, bill amounts and payment amounts.
- Known limits: one country, one era, one bank's book. Class balance
  is skewed (about 22 percent positive). The data carries no names or
  account numbers, but it is still treated as sensitive-shaped:
  public data only, never real client data, in every build here.
- Licence and terms: UCI repository terms; cite Yeh and Lien (2009).
