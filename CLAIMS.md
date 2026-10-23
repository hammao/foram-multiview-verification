# Claims and where to check them

Every quantitative statement in the revised AIIG-D-26-00130 is recomputed by `python verify_claims.py` from the
files in `data/` and `results/`. The expected values in `expected_values.json` are the values printed in the
manuscript; the script re-derives each one from the raw prediction files and fails on any disagreement.

## Partition (Section 2.4, Table 1)

- 50 species, 140 specimens, 1683 images; multiplicity profile {'1': 4, '2': 13, '3': 26, '4': 4, '5': 2, '6': 1}.
- Tiers: {"train": {"n_images": 1044, "n_species": 50, "n_specimens": 87, "sources": {"gif_original": 522, "pmax_tiff": 384, "stacked_tiff": 138}}, "val": {"n_images": 243, "n_species": 20, "n_specimens": 20, "sources": {"gif_original": 120, "pmax_tiff": 120, "stacked_tiff": 3}}, "test": {"n_images": 396, "n_species": 33, "n_specimens": 33, "sources": {"gif_original": 198, "pmax_tiff": 198}}}.
- No specimen appears in two tiers; every test species has at least two training specimens; the SHA-256 of each
  partition file matches `split_provenance.json` (frozen before training).
- The external cohort (39 images) shares no file or checksum with the catalogue (`audit_independence.json`).

## Held-out individuals (Section 3.3, Table 3)

- single_stream: image top-1 0.876 (347/396), genus 0.907, specimen majority 0.909 (30/33), macro-F1 0.877, ECE 0.238 → 0.115 (T = 0.796); deployed epoch 28.
  External cohort: top-1 0.590, top-3 0.872 (n = 39).
- dual_stream: image top-1 0.854 (338/396), genus 0.891, specimen majority 0.909 (30/33), macro-F1 0.858, ECE 0.219 → 0.117 (T = 0.830); deployed epoch 22.
  External cohort: top-1 0.641, top-3 0.795 (n = 39).

## Firewall (Section 2.9)

- `results/<arm>/selection_provenance.json` records `audit_cohort_used: false`, `test_split_used: false`;
  `run_status.json` records all 40 epochs run and the selected epoch.
- `results/<arm>/temperature.json` records the split the temperature was fitted on (`val`).
