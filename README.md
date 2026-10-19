# ModelGate: a training pipeline with a gate that can block a bad model

Week 1 of the PROOF roadmap (Pattern 19, Episode 2), live at
[jesunahmadushno.com/roadmap/tangerine-ai-engineer](https://jesunahmadushno.com/roadmap/tangerine-ai-engineer/).
The roadmap reads
one real job post as a test list; this build answers its first line, the
one about engineering delivery of an AI platform: training and
deployment pipelines, monitoring, and governance. So the pipeline is
not the point. The gate is.

## What it does

1. `src/train.py` trains a small credit-risk scorer (logistic
   regression) on the public UCI Default of Credit Card Clients
   dataset, freezes a stratified test split, hashes the frozen file,
   and writes a model card. It writes no evidence records.
2. `src/evaluate.py` scores the model on the frozen test split only
   and writes an eval record that carries the test file's hash.
3. `src/data_card.py` writes the data card from the dataset on disk.
4. `src/gate.py` is the promotion gate. It promotes the model only if
   all three hold:
   - an eval record exists, its AUC beats the floor set before
     training (0.65, in `gate_config.json`), and its hash matches the
     frozen test set;
   - a data card exists;
   - an approval line (`APPROVED <name> <date>`) sits in
     `records/approvals.log`.

The check order is read, decide, report. Nothing is copied or promoted
before the decision, because a gate that runs after the copy step is a
decoration.

## The done test

Run `./run_demo.sh`. It trains, then runs the gate twice, and saves
both outputs under `demo/`.

Run 1, records incomplete (actual output, `demo/run1_blocked.txt`):

```
ModelGate promotion check
model: model v1
[1/3] eval record ... MISSING (records/eval_record.json not found)
[2/3] data card ... MISSING (records/data_card.md not found)
[3/3] approval line ... MISSING (no APPROVED line in records/approvals.log)
PROMOTE model v1: BLOCKED
reasons: no eval record; no data card; no approval line
```

Then the eval and data card steps run, one approval line is added, and
Run 2 (actual output, `demo/run2_pass.txt`):

```
ModelGate promotion check
model: model v1
[1/3] eval record ... OK (AUC 0.724 >= floor 0.65, 6000 frozen rows, hash verified)
[2/3] data card ... OK (records/data_card.md)
[3/3] approval line ... OK (Jesun Ahmad Ushno, 2026-10-19)
PROMOTE model v1: PASS
```

The full records live in `records/eval_record.json`, and the two files
in `demo/` are the saved terminal outputs of real runs of this code.

## Run it yourself

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./run_demo.sh
```

`src/get_data.py` downloads the dataset from the UCI Machine Learning
Repository (public, 2005, Taiwan card clients) and normalizes it to
`data/credit.csv`. Public data only: never point these builds at real
client data.

## Why a gate at all

A demo is a claim. A gate log is proof. Every model this pipeline ever
ships carries the three records a reviewer can check: the score, the
data it came from, and the human who approved it. If it is not logged,
it did not happen.
