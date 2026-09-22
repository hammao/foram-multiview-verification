# The evaluation firewall

The single most important methodological property of this study is that the 45-image Leica audit
cohort never influenced any modelling decision. It was scored once, after everything else had been
frozen. This document explains how that is enforced, how to check it, and how an earlier version of
the pipeline violated it.

## What is fixed on which split

| Decision | Fixed using | Never uses |
|---|---|---|
| Which epoch's checkpoint is deployed | validation split | test split, audit cohort |
| Calibration temperature | validation split | test split, audit cohort |
| Confidence thresholds for the coverage curve | fixed a priori (0.2, 0.5, 0.9) | audit cohort |
| Reported held-out-view metrics | test split | audit cohort |
| Reported out-of-distribution metrics | audit cohort, scored once | — |

The validation split is a sixth of the views and is the only split that touches model selection.
The test split is reported but never optimised against. The audit cohort is reported once.

## How selection actually happens

`scripts/select_best_checkpoint_by_val.py` scores every saved epoch checkpoint on the validation
split, writes the full table to `checkpoint_selection_val.csv`, takes the argmax of validation
accuracy, and records what it did in `selection_provenance.json`:

```json
{
  "selected_epoch": 2,
  "selection_metric": "val_acc",
  "selection_split": "val",
  "selection_value": 0.9653465346534653,
  "n_candidates": 7,
  "audit_cohort_used": false,
  "test_split_used": false
}
```

Both files are shipped, for both arms, so the selection is auditable rather than asserted. The
single-stream arm deployed epoch 2 and the dual-stream arm deployed epoch 3.

## How to check it yourself

```bash
python verify_claims.py --group firewall
```

This does not just read back the `audit_cohort_used: false` flag, which would be circular. It
recomputes the argmax of validation accuracy across all candidate epochs in
`checkpoint_selection_val.csv` and asserts that the deployed epoch equals it. If the deployed
checkpoint had been chosen to look good on the audit cohort, it would almost certainly not be the
validation argmax, and the check would fail.

Two further things are worth confirming by hand. The temperature in
`results/<arm>/temperature.json` carries `"split": "val"`, and the same value appears as
`temperature` in `eval_report.json`, so the scalar fitted on validation is the one applied to the
test split without refitting. And the three thresholds annotated on the coverage curve are
hard-coded in `scripts/plot_coverage_accuracy.py` rather than chosen after seeing the curve.

## The retired selector, and why it is not in this repository

An earlier iteration of this pipeline contained `select_best_checkpoint_by_verified.py`, which
picked the deployed checkpoint by scoring candidate epochs against the human-verified cohort. That
is a leak. Choosing a checkpoint by its performance on the cohort turns the cohort into a
validation set, and any accuracy subsequently reported on it is optimistically biased by an amount
that cannot be estimated after the fact.

The script was removed from the pipeline, all affected numbers were regenerated under
validation-only selection, and the reported audit accuracy fell as a result. The audit figures in
the manuscript, 42.2% top-1 for the single-stream arm and 51.1% for the dual-stream arm, are the
post-fix numbers.

The retired script is deliberately not included in this repository. Shipping it would invite
someone to run it and regenerate a leaked result that contradicts the paper. What matters for
verification is the selector that was actually used, `select_best_checkpoint_by_val.py`, which is
included, together with the provenance files that record its decisions.

## Why the audit cohort is small, and why that is stated rather than fixed

Forty-five images is not many, and the Wilson interval on 42.2% runs from 29.0% to 56.7%. The
cohort is small because every image in it was independently verified by an expert against the
morphological criteria in Section 2, which is slow work.

The cohort could have been enlarged by relabelling model predictions, but predictions the model
already gets right are the ones a human is most likely to wave through, so the cohort would drift
toward being easy. It was instead fixed in advance, stratified into normal and difficult strata
(29 and 16 images), and scored once. The paper reports the interval alongside the point estimate
and does not claim a dual-stream advantage on this cohort, because the two intervals overlap
substantially.
