# Reviewer verification materials

### *Improving Foraminifera Identification through Taxonomically Informed Deep Learning* (AIIG-D-26-00130)

This repository lets a reader recompute every number in the manuscript. It ships the saved model
outputs, the frozen data splits, the evaluation images and the code, so the claims can be checked
without a GPU and without re-running training. The two model checkpoints are published as release
assets for anyone who wants to regenerate the outputs from the weights themselves.

## Check the claims in two minutes

```bash
git clone https://github.com/hammao/foram-multiview-verification.git
cd foram-multiview-verification
pip install pandas
python verify_claims.py
```

This runs 121 checks and prints a pass or fail for each. No GPU, no model weights, no network
access. Every expected value is hard-coded from the manuscript text, so the script tests the paper
against the data rather than reporting what the data happens to say. It exits non-zero on any
disagreement.

[CLAIMS.md](CLAIMS.md) lists every claim, the file it comes from, and the check that covers it.

## Reproduce the outputs from the released weights

If you would rather regenerate the prediction files instead of trusting the ones shipped here:

```bash
pip install -r requirements.txt
./reproduce_eval.sh
```

This downloads both checkpoints from the latest release, verifies their SHA-256 sums, re-runs
evaluation on the test split and the audit cohort, and diffs the fresh outputs against the
committed ones. It needs about 2 GB of disk and runs in a few minutes on a GPU or roughly half an
hour on CPU.

## The two accuracy figures, and why they differ

The paper reports 99.0% on the held-out-view test split and 42.2% on the locked audit cohort. Both
are real and they measure different things.

The **held-out-view test split** holds out one of the six rotational views of each specimen. The
same physical specimens appear in training, so the figure measures recognition of a familiar
individual from a new angle. `verify_claims.py --group dataset` confirms this directly: all 32
specimens in the test split also appear in training. This is the number that most single-instance
foraminifera classification studies implicitly report, and it is why the paper argues such numbers
should not be read as generalisation.

The **locked Leica audit cohort** is 45 separately acquired, expert-verified images fixed before
any model saw them. This is the out-of-distribution figure, and the honest one. The gap between
the two is the main empirical result of the paper.

The clearest single illustration is *Hoeglundina elegans*, which scores a perfect F1 under
held-out views and drops to 0.738 accuracy when its specimen is held out entirely
(`verify_claims.py --group helegans`).

## Evaluation firewall

The audit cohort never influenced checkpoint selection, the calibration temperature, or any
threshold. It was scored once, after everything else was frozen. The selection provenance is
shipped for both arms and `verify_claims.py --group firewall` recomputes the validation argmax
rather than trusting the recorded flag.

An earlier version of this pipeline selected checkpoints against the audit cohort, which is a leak.
That script was retired, the affected numbers were regenerated, and the reported audit accuracy
fell. [docs/EVALUATION_FIREWALL.md](docs/EVALUATION_FIREWALL.md) documents what happened and why
the retired selector is deliberately not shipped here.

## What is in the repository

```
verify_claims.py         121 checks of the manuscript's numbers against the saved outputs
CLAIMS.md                claim-by-claim map from the paper to the files and checks
reproduce_eval.sh        re-runs evaluation from the released weights and diffs the results
compare_reproduced.py    the diff step, comparing fresh outputs against the committed ones

results/<arm>/           saved outputs for single_stream and dual_stream
  eval_report.json           test-split metrics, confusion matrix, calibration
  test_predictions.csv       per-image predictions on the 202-image test split
  audit_predictions.csv      per-image top-3 predictions on the 45-image audit cohort
  audit_summary.json         pooled and per-stratum audit accuracy with Wilson intervals
  temperature.json           temperature fitted on validation, with before/after ECE and NLL
  training_metrics.csv       per-epoch losses and accuracies
  checkpoint_selection_val.csv   validation accuracy of every candidate epoch
  selection_provenance.json      which epoch was deployed and on what basis
  noise_results.csv          per-image predictions under Gaussian and salt-and-pepper corruption
  rarity_f1_vs_ntrain.csv    per-species F1 against training-set frequency

data/
  manifest/splits/         the frozen holdout_views split at seed 1337
  eval_images/             the 404 validation and test images
  audit_images/            the 45 audit cohort images
  audit/                   audit labels with difficulty strata

configs/                 training and evaluation configuration for both arms
src/, scripts/           the model, dataset, training and evaluation code
docs/                    evaluation firewall, data accounting, split checksums
CHECKSUMS.sha256         SHA-256 of every shipped file
```

The 8,219-image acquisition catalogue and the raw specimen images are not included here; they are
too large for a git repository and will be deposited with the accepted manuscript. What is included
is everything needed to verify the claims: the 5,967-image split manifest, the 449 images actually
used for evaluation, and every saved model output.
[docs/DATA_ACCOUNTING.md](docs/DATA_ACCOUNTING.md) reconciles 8,219 with 5,967.

## Model checkpoints

Both checkpoints are attached to the latest [release](https://github.com/hammao/foram-multiview-verification/releases/latest),
roughly 350 MB each, with SHA-256 sums in the release notes. Both are ConvNeXt-Base
(`convnext_base.fb_in22k_ft_in1k`) fine-tuned at 384x384 over 31 species. The single-stream
checkpoint is epoch 2 and the dual-stream checkpoint is epoch 3, both chosen by validation
accuracy.

## Licence

Code is MIT (see [LICENSE](LICENSE)). The data files, images, manifests and model outputs are
released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use either,
please cite the paper; see [CITATION.cff](CITATION.cff).

## Contact

Questions about verification, including failed checks, are welcome as GitHub issues.
