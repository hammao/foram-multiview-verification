# Claims and where to check them

Every quantitative statement in AIIG-D-26-00130 is listed below with the file it comes from and
the automated check that recomputes it. Running

```bash
python verify_claims.py
```

executes all 121 checks and exits non-zero if the paper and the data disagree. The expected
values are hard-coded in `verify_claims.py` from the manuscript text, so the script tests the
paper rather than describing it. Use `--group` to run one section at a time, for example
`python verify_claims.py --group audit`.

Nothing here needs a GPU, the model weights, or network access. Every number is recomputed from
the prediction and metric files in `results/`. The released checkpoints are only needed if you
want to regenerate those files yourself, which `reproduce_eval.sh` does.

## A note on what the two evaluation sets mean

The paper reports two very different accuracy figures and the gap between them is the point
rather than an inconsistency.

The **held-out-view test split** (202 images, 99.0% top-1) holds out one of six views per
specimen. The same physical specimens appear in training, so this number measures recognition of
familiar individuals under a new rotation. It is not a generalisation estimate, and
`verify_claims.py --group dataset` confirms the overlap explicitly: all 32 specimens in the test
split also appear in training.

The **locked Leica audit cohort** (45 images, 42.2% top-1) consists of separately acquired,
human-verified specimens that were fixed before any model touched them. This is the honest
out-of-distribution figure. Checkpoint selection, the calibration temperature and every threshold
were fixed on the validation split before the cohort was scored, which
`verify_claims.py --group firewall` verifies against the recorded selection provenance.

## Dataset and splits (Section 3.1)

| Claim | Value | Source | Check group |
|---|---|---|---|
| Images acquired | 8,219 | acquisition log | `dataset` |
| Images entering the modelling partitions | 5,967 | `data/manifest/splits/*.csv` | `dataset` |
| Images excluded during curation | 2,252 | difference of the two above | `dataset` |
| Train / validation / test | 5,563 / 202 / 202 | `data/manifest/splits/*.csv` | `dataset` |
| Species | 31 | `data/manifest/splits/test.csv` | `dataset` |
| No image appears in two partitions | 0 shared | `data/manifest/splits/*.csv` | `dataset` |
| Specimens recur across partitions (view-level split) | 32 shared | `data/manifest/splits/*.csv` | `dataset` |

## Held-out-view test split (Section 3.2, Table 3)

| Claim | Single-stream | Dual-stream | Source | Check group |
|---|---|---|---|---|
| Top-1 accuracy | 0.990 | 0.916 | `results/<arm>/eval_report.json` | `test` |
| Macro F1 | 0.991 | 0.909 | `results/<arm>/eval_report.json` | `test` |
| Weighted F1 | 0.990 | 0.904 | `results/<arm>/eval_report.json` | `test` |
| Correct / misclassified | 200 / 2 | 185 / 17 | `results/<arm>/test_predictions.csv` | `test` |

Accuracy is recomputed from the per-image predictions as well as read from the report, so the two
must agree. The two single-stream errors named in the Figure 5 caption are checked by name:
*Quinqueloculina pulchella* predicted as *Q. crassicarinata*, and *Spiroloculina communis*
predicted as *Triloculina plicata*.

## Calibration (Section 3.2, Table 3)

The temperature is fitted by LBFGS on the validation split only, then applied unchanged to the
test split. ECE uses 15 equal-width confidence bins.

| Claim | Single-stream | Dual-stream | Source | Check group |
|---|---|---|---|---|
| Fitted temperature | 0.546 | 0.658 | `results/<arm>/temperature.json` | `calibration` |
| Validation ECE, before → after | 0.145 → 0.028 | 0.133 → 0.018 | `results/<arm>/temperature.json` | `calibration` |
| Validation NLL, before → after | 0.243 → 0.129 | 0.273 → 0.188 | `results/<arm>/temperature.json` | `calibration` |
| Test ECE, before → after | 0.178 → 0.044 | 0.109 → 0.049 | `results/<arm>/eval_report.json` | `calibration` |
| Test NLL, before → after | 0.230 → 0.069 | 0.455 → 0.447 | `results/<arm>/eval_report.json` | `calibration` |

The ordering reverses under scaling: the dual-stream arm is better calibrated before the scalar is
fitted (0.109 against 0.178) and worse after it (0.049 against 0.044). Both arms were calibrated
identically, so the comparison in Table 3 is like for like.

## Locked Leica audit cohort (Section 3.5)

| Claim | Single-stream | Dual-stream | Source | Check group |
|---|---|---|---|---|
| Cohort size | 45 | 45 | `data/audit/verified_labels_audit_v2.csv` | `audit` |
| Top-1 | 0.422 (19/45) | 0.511 (23/45) | `results/<arm>/audit_predictions.csv` | `audit` |
| Top-1 95% Wilson interval | 0.290–0.567 | 0.370–0.650 | recomputed in the checker | `audit` |
| Top-3 | 0.778 (35/45) | 0.756 (34/45) | `results/<arm>/audit_predictions.csv` | `audit` |
| Normal stratum, n=29 | 0.448 | 0.517 | `results/<arm>/audit_summary.json` | `audit` |
| Difficult stratum, n=16 | 0.375 | 0.500 | `results/<arm>/audit_summary.json` | `audit` |

The Wilson intervals are recomputed from the raw hit counts rather than copied from the summary
files. They overlap substantially between the two arms, which is why the paper does not claim a
dual-stream advantage on this cohort despite the higher point estimate.

## Coverage-accuracy behaviour (Figure 6)

Single-stream, calibrated, on the audit cohort.

| Confidence threshold | Images covered | Accuracy on covered | Check group |
|---|---|---|---|
| 0.2 | 45 of 45 | 0.422 | `coverage` |
| 0.5 | 36 of 45 | 0.528 | `coverage` |
| 0.9 | 15 of 45 | 0.733 | `coverage` |

Source: `results/single_stream/audit_confidence_summary.json`, cross-checked against
`audit_confidence_predictions.csv`. The checker also asserts that accuracy rises monotonically
across the whole curve, which is the property the triage argument in Section 3.5 depends on.

## Robustness under corruption (Figure 8)

Single-stream, 202 test images per level.

| Corruption | Accuracy | Check group |
|---|---|---|
| Clean | 0.970 | `noise` |
| Gaussian σ=5 | 0.866 | `noise` |
| Gaussian σ=10 | 0.748 | `noise` |
| Gaussian σ=25 | 0.614 | `noise` |
| Gaussian σ=50 | 0.510 | `noise` |
| Salt-and-pepper 1% | 0.916 | `noise` |
| Salt-and-pepper 3% | 0.876 | `noise` |
| Salt-and-pepper 7% | 0.738 | `noise` |

Source: `results/single_stream/noise_results.csv`, which holds one row per image per corruption
level. The rendered corrupted images are not shipped because they run to several hundred
megabytes; `scripts/noise_tests.py` regenerates them.

## Rarity analysis (Section 3.4)

| Claim | Value | Check group |
|---|---|---|
| Species scored | 31 | `rarity` |
| Species not classified perfectly | 4 | `rarity` |
| Every species with fewer than 180 training images is error-free | true | `rarity` |
| Training counts of the four imperfect species | 180 to 345 | `rarity` |
| *Bigenerina nodosaria* test support | 1 image | `rarity` |

Source: `results/single_stream/rarity_f1_vs_ntrain.csv`. The last row is included deliberately:
*B. nodosaria* is perfectly classified on a single test image, which is too thin to support any
claim about rare taxa, and Section 3.4 says so.

## Training dynamics (Figure 4, Section 3.2)

| Claim | Value | Check group |
|---|---|---|
| Single-stream epochs before early stopping | 7 | `training` |
| Single-stream training loss, epoch 1 → 7 | 1.50 → 0.99 | `training` |
| Single-stream training accuracy, epoch 1 → 7 | 0.760 → 0.935 | `training` |
| Single-stream peak validation accuracy | 0.965 at epoch 2 | `training` |
| Dual-stream peak validation accuracy | 0.965 at epoch 3 | `training` |
| Dual-stream teacher loss at epoch 1 | 0.93 | `training` |
| Dual-stream student loss at epoch 1 | 1.45 | `training` |

Source: `results/<arm>/training_metrics.csv`. The teacher and student losses at epoch 1 support
the explanation in Section 3.2 for why the dual-stream arm underperforms: the specimen-level
teacher fits faster than the view-level student from the very first epoch, so the distillation
term pulls the student toward a head that has already begun to overfit.

## Evaluation firewall (Section 3.3)

| Claim | Value | Check group |
|---|---|---|
| Selection metric | validation accuracy | `firewall` |
| Audit cohort used for selection | no | `firewall` |
| Test split used for selection | no | `firewall` |
| Single-stream deployed epoch | 2 | `firewall` |
| Dual-stream deployed epoch | 3 | `firewall` |

Source: `results/<arm>/selection_provenance.json` and `checkpoint_selection_val.csv`. The checker
does not simply read the recorded epoch; it recomputes the argmax of validation accuracy over all
candidate checkpoints and asserts the deployed epoch equals it. See
[docs/EVALUATION_FIREWALL.md](docs/EVALUATION_FIREWALL.md) for how this replaced an earlier
selection procedure that did read the audit cohort.

## Hoeglundina elegans probe (Section 3.6)

| Claim | Value | Check group |
|---|---|---|
| Specimens in the dataset | 2 | `helegans` |
| Held-out-view F1 | 1.000 | `helegans` |
| Accuracy holding out specimen 0 | 1.000 (6 images) | `helegans` |
| Accuracy holding out specimen 28 | 0.738 (145 images) | `helegans` |

Source: `results/single_stream/hoeglundina_elegans_probe.json`. This is the clearest single
illustration of the view-level split's optimism: a species that is perfect under held-out views
drops to 0.738 when the model has never seen the specimen, and the two folds disagree sharply
because one of them holds out only six images.

## Claims not covered by this repository

Two kinds of statement cannot be checked here.

Taxonomic identifications rest on expert morphological assessment and WoRMS harmonisation, not on
computation. The accepted names, authorities and AphiaIDs appear in the manuscript's Table 1 and
can be checked against [WoRMS](https://www.marinespecies.org/) directly.

Statements about acquisition hardware, imaging protocol and specimen provenance are described in
Section 2 and are not derivable from the released outputs.
