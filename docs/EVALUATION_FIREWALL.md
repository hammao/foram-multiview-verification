# Evaluation firewall

Which data may influence which decision. Written in response to reviewer comments M2 and
M17, which are correct.

## The rule

| Data | May influence | May not influence |
|---|---|---|
| `splits/train.csv` (5,563 images) | Weight updates | — |
| `splits/val.csv` (202 images) | Checkpoint selection, temperature calibration, early stopping | — |
| `splits/test.csv` (202 images) | Nothing. Reported once. | Any choice of model, epoch, threshold or temperature |
| `verified/verified_labels_audit_v2.csv` (45 images) | Nothing. Reported once. | Any choice of model, epoch, threshold or temperature |

## What went wrong

`scripts/select_best_checkpoint_by_verified.py` ranked every epoch checkpoint by top-1
accuracy on the audit labels:

```python
res = pd.DataFrame(rows).sort_values(["top1", "top3", "epoch"], ascending=[False, False, True])
best = res.iloc[0].to_dict()
...
(dst / "classifier.pt").write_bytes(best_path.read_bytes())
```

The `--apply-to` path then copied that winner over the deployed `classifier.pt`. Any
accuracy subsequently reported on the audit cohort was therefore a selection statistic:
the cohort had chosen the model, so its score was the maximum over candidates rather than
an unbiased estimate of generalisation. With 12 candidate epochs and 45 images, the upward
bias is substantial — the difference between best and median epoch on that cohort spans
several percentage points, all of it selection.

This is a single concrete defect with a file and a line number. It is conceded rather than
argued.

## What replaced it

`scripts/select_best_checkpoint_by_val.py` ranks the same epoch checkpoints on
`splits/val.csv` and writes `selection_provenance.json` alongside the deployed checkpoint,
recording the selection split, the metric, the value and explicit
`"audit_cohort_used": false` / `"test_split_used": false` fields. `--split` accepts only
`val`; test and audit are not reachable options.

Ties are broken toward the earlier epoch, so the least-trained model achieving the best
validation score is preferred.

The old script now refuses to run without `--diagnostic-only`, and `--diagnostic-only`
forbids `--apply-to`. It no longer prints which epoch scored best, because quoting that
number would reintroduce the same selection by hand. It survives only to describe how
audit accuracy varies across epochs *after* a model has been selected on validation.

## Expected consequence

Selecting on validation rather than the audit cohort will lower reported audit accuracy.
The pre-revision sweep records put best audit top-1 at 0.600 on the 45-image cohort against
0.970 internal, and the best epoch was chosen on the audit cohort itself, so the honest
figure will sit below 0.600. That drop is the correction, not a regression.

## Drive140 (50 species / 140 specimens) — the rule set in force from 16 Oct 2026

The 31-species rows above are retained for the record. The paper runs under
`configs/train_{single,dual}_stream_drive140_paper.yaml` use the frozen specimen partition in
`processed/drive140/manifest/splits_specimen_tvt/` (files chmod 0444; SHA-256 of every file in
`split_provenance.json`, frozen 2026-10-16T13:52:43Z, before any paper training started).

| Data | Unit | May influence | May not influence | Scored |
|---|---|---|---|---|
| `splits_specimen_tvt/train.csv` (1,044 images, 87 shells, 50 spp) | specimen | Weight updates | — | every epoch |
| `splits_specimen_tvt/val.csv` (243 images, 20 shells, 20 spp) | specimen | Checkpoint epoch (earliest best val acc), temperature, gating threshold, every recipe choice in exploratory runs | — | every epoch |
| `splits_specimen_tvt/test.csv` (396 images, 33 shells, 33 spp) | specimen | Nothing | Any choice of model, epoch, threshold, temperature or hyperparameter | once, after deploy |
| `splits_specimen_tvt/audit_overlap_39.csv` (39 Leica images, 18 spp) | image | Nothing | Same | once, after deploy |

Rules that make the table verifiable from files rather than prose:

- The partition is a function of multiplicity only (N ≥ 4: highest ID test, second-highest val;
  N = 3: highest test; N = 2: highest val; N = 1: train), assignment deterministic by specimen
  ID. `split_provenance.json` records the rule text, the three priorities, the rejected
  alternative and the per-species assignment.
- `early_stopping_patience: 0`; all 40 epochs run; `deploy_best_validation_checkpoint` writes
  `selection_provenance.json` with `audit_cohort_used: false`, `test_split_used: false`. The
  trainer never reads `test.csv`.
- `audit_independence.json`: SHA-256 of the 39 audit files against 983 Drive source files and
  1,683 resized PNGs — 0 matches. The audit shells are the same individuals as catalogue shells,
  re-imaged on a Leica at orientations outside the six training rotations; the cohort therefore
  measures platform and orientation shift on known individuals and is never described as new
  specimens. Which split each audited individual sits in is listed per row for transparency.
- `scripts/run_paper_eval_battery.py` refuses to run on a checkpoint whose `run_status.json`
  is not `complete`, re-verifies the frozen file hashes, fits temperature on `val` only, and
  scores test and audit once (`eval_audit_once.py` once-marker).
- The three open-nomenclature OTUs (*Cibicides* sp. SdM-S057, *Cibicidoides* sp. SdM-S049,
  *Quinqueloculina* sp. SdM-S094) have N = 2 and therefore sit in validation only; the test
  tier contains binomial species exclusively.
- Headline figures come from `test.csv` (image top-1, genus top-1, specimen majority over the
  12 images of each shell). Validation figures are reported as validation. The old
  `splits_specimen/` (train/val only, used for the exploratory 0.83 run) is not a paper split.
