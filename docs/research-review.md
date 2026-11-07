# Research review: ModelGate (Pattern 19, week 1)

A self-review of this repo against a PhD-style research rubric, plus the
engineering rubric that fits a governance tool. Grades are A (meets the
bar), B (partly), C (weak), D (missing). Each remaining gap names the
exact compute or data it would take to close.

Numbers marked *computed* come from `site/src/model.js` run on the
in-repo CSVs (the same code the live dashboard runs, checked against the
recorded sklearn result by `site/src/model.check.mjs` at every build).
Recorded numbers come from `records/eval_record.json`.

## Summary

ModelGate is an engineering build, not a study: its claim is that the
promotion gate can say no. That claim is now tested (16 behaviour tests in
`tests/test_gate.py`, run in CI before the dashboard builds) and two real
defects in the gate were fixed on the way. The model inside the gate is a
teaching baseline whose evaluation is thin; the dashboard adds the
uncertainty and baseline the records lack, labelled as computed there.

| Criterion | Grade | Notes |
|---|---|---|
| Question | A | One falsifiable engineering question: does the gate block when any record is missing, faked, tampered or below the floor? |
| Hypotheses | B | The gate's pass/block rules are stated in `src/gate.py` and README; the model has no hypothesis beyond "beats a 0.65 AUC floor". |
| Method | B | Read, decide, report order; hashed frozen split; floor fixed before training. Single split and single seed. |
| Baselines | C | No baseline in the records. Computed: always predicting "no default" scores 77.9% on the frozen rows against the recorded 81.1%. |
| Statistics and uncertainty | C | No interval in the records. Computed: 95% percentile bootstrap of the test AUC, 500 resamples, seed 19: 0.705 to 0.740. |
| Reproducibility | B | `requirements.txt` now pinned; a fresh venv reproduced AUC 0.724, accuracy 0.811 and the frozen sha256 exactly. Records carry `date.today()`, so reruns are not byte-identical. |
| Threats to validity | B | Listed on the dashboard and below; model card now states that ID and protected attributes are inputs. |
| Related work | D | Only "cite Yeh and Lien (2009)". |
| Limitations | B | Model card and data card state one country, one era, teaching use only. |
| Engineering rigor | B | Tests and CI now exist; model artifact and approval are still not bound to the eval record. |

## Fixed in this pass (cheap)

1. **No tests, no CI.** Added `tests/test_gate.py` (stdlib only, under a
   second) and a CI step in `.github/workflows/pages.yml`.
2. **Floor read from the eval record.** `check_eval` used
   `rec.get("floor_auc")`, so editing the eval record could lower the bar,
   contradicting `gate_config.json`'s own note. The gate now reads the
   floor from `gate_config.json`. Written red first, then fixed; the
   shipped output is unchanged, so `demo/` stays valid.
3. **Crash instead of BLOCKED.** A missing or non-numeric `auc` or floor
   raised a `TypeError`. It now blocks with a reason.
4. **Data card omitted ID.** "Columns: 25 (23 features + target)" now reads
   "(ID + 23 features + target)", in both the card and `src/data_card.py`.
5. **Inputs not disclosed.** The model card (and the template in
   `src/train.py`) now says ID, SEX, MARRIAGE and AGE are inputs and that no
   subgroup check has been run in the repo.
6. **Unpinned stack.** Exact versions recorded after a successful
   reproduction (Python 3.12.3, scikit-learn 1.9.1, pandas 3.1.1,
   numpy 2.5.4).
7. **Browser port proven identical.** The site build fails unless the JS
   gate prints exactly `demo/run1_blocked.txt` and `demo/run2_pass.txt`,
   and the browser refit matches the recorded AUC and accuracy within
   0.001 (computed refit AUC 0.7236 against recorded sklearn 0.724).

## Computed audit results (not in the repo's records)

- **Ablation** (refit without the feature, frozen-split AUC): all 24
  features 0.7236; without ID 0.7235; without SEX 0.7231; without
  MARRIAGE 0.7238; without AGE 0.7237; without all four 0.7245. Every
  change is far inside the bootstrap half-width (about 0.018), so on this
  split the four columns buy nothing measurable.
- **Subgroups by SEX** (UCI coding 1 = male, 2 = female), threshold 0.5:
  SEX 1, 2,408 rows, AUC 0.730, flagged 9.2%, recall 27.2%; SEX 2,
  3,592 rows, AUC 0.715, flagged 5.2%, recall 18.9%. These are point
  estimates from one split with no interval; they describe, they do not
  establish a disparity.

## Remaining gaps and what each needs

| Gap | What closes it | Cost |
|---|---|---|
| Eval record not bound to `models/model_v1/model.pkl` | Write the model file's sha256 into the eval record in `src/evaluate.py` and check it in `src/gate.py`. Changes the record format, so the shipped eval record and `demo/run2_pass.txt` must be regenerated: an Owner decision. | CPU seconds, one rerun of `run_demo.sh` in the pinned venv |
| Approval not bound to model version or date window | Require `APPROVED <name> <date> <model_version>` and check the version; changes `records/approvals.log` format and the transcripts. Owner decision. | Minutes, no compute |
| ID and protected attributes as inputs | Train a model v2 without them, give it its own eval record and approval, and let it pass or fail the gate. The computed ablation suggests no AUC cost. Owner decision, since it changes the recorded 0.724. | CPU seconds |
| Single split, no interval in the records | Add repeated stratified k-fold (for example 5x2) and a bootstrap interval to `src/evaluate.py`, recorded alongside the frozen-split AUC. | Under a minute of CPU |
| No calibration, no PR-AUC, threshold 0.5 unexplained | Add a reliability curve, Brier score and PR-AUC to the eval record; choose the threshold from a stated cost ratio. | Seconds of CPU; the cost ratio is a business input |
| Fairness not assessed with uncertainty | Bootstrap the per-group AUC, flag rate and recall gaps; add a gate check that blocks if a gap exceeds a pre-set limit. The limit is a policy decision. | Seconds of CPU |
| Records not byte-reproducible | Take the evaluation date from `gate_config.json` or an environment variable instead of `date.today()`. | Minutes |
| Data provenance hash truncated | Store the full sha256 of `data/credit.csv` and of the UCI source file in the data card and verify it in `src/get_data.py`. | Minutes, one download |
| Related work | Compare with published results on this dataset (Yeh and Lien, 2009, and later benchmarks) and with governance practice: model cards (Mitchell et al., 2019), datasheets (Gebru et al., 2021) and the ML Test Score (Breck et al., 2017). | Reading time only |

## Threats to validity

- One public 2005 dataset from one bank in one country; nothing here
  generalizes to current applicants. The cards say so.
- The frozen split is stratified with one seed; the AUC could move by
  about the bootstrap half-width on a different split.
- The gate trusts the eval record's AUC; it re-verifies the test file's
  hash but does not re-score the model.
- The browser refit uses Newton's method; sklearn's lbfgs stops at its
  tolerance, which is why the computed and recorded AUC differ in the
  fourth decimal. The gate always uses the recorded number.
