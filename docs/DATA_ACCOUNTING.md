# Data accounting: every image, and why it is or is not used

Three numbers describe the image collection and they are easy to conflate. This document
reconciles them exactly, so the denominator behind every reported accuracy is unambiguous.

| Count | What it is |
|---|---|
| **10,496** | PNG files present in `processed/resized/` |
| **8,219** | rows in `processed/manifest/views_resized.csv` — the current dataset |
| **5,967** | images entering the modelling splits — **the denominator for every reported metric** |

Verified against the regenerated `holdout_views` split at seed 1337. Every manifest row resolves
to a file on disk (0 missing), so the discrepancies are all in one direction: files and rows that
are *not* used.

## 8,219 → 5,967: the 2,252 unassigned images

Split composition:

| Split | Images | Held-out views |
|---|---|---|
| train | 5,563 | — |
| val | 202 | 32 |
| test | 202 | 32 |
| **total** | **5,967** | |

The 2,252 manifest rows outside the splits are **all `zstack_slice`**, and every one belongs to a
view assigned to validation (1,063) or test (1,189). They are the slices beyond the per-view cap:
`max_z_slices_per_view_in_val` and `..._in_test` are both 5, and the cap is enforced exactly —
every held-out view contributes precisely 5 z-stack slices, no more and no fewer.

Note that the cap applies to z-stack slices only, not to all images. Each held-out view also
contributes its uncapped extended-depth-of-field composite and GIF-derived frame, so val and test
contain 150 z-slices + 29 GIF frames + 23 composites = 202 images each. Validation and test are
exactly symmetric in composition.

The cap exists so that a single view with a deep z-stack cannot dominate the held-out metrics.
Without it, one specimen contributing 128 slices would outweigh twenty specimens contributing five.
Training is deliberately uncapped, since there the redundancy is useful augmentation.

## 10,496 → 8,219: the 2,277 orphan files

These are files left on disk by earlier pipeline runs that the current manifest does not
reference. They are inert — nothing reads them — but they are the reason a naive file count
disagrees with the dataset size. Two distinct causes:

### 1,966 files under superseded species names

Written before the taxonomy was harmonised, and never cleaned up when the folders were renamed:

| Legacy name on the files | Current accepted name | Files |
|---|---|---|
| *Adelosina pulchella* | *Quinqueloculina pulchella* | 360 |
| *Agglutinella arenata* | *Siphonaperta arenata* | 332 |
| *Pyrgo anomala* | *Pyrgo inornata* | 319 |
| *Textularia kerimbaensis* | *Sahulia kerimbaensis* | 312 |
| *Quinqueloculina limbata* | *Pseudotriloculina limbata* | 265 |
| *Ammobaculites reophaciformis* | *Nodulina dentaliniformis* | 216 |
| *Elphidium clavatum* | *Cribroelphidium clavatum* | 162 |

Every mapping is recorded in `configs/species_synonyms.csv`. Two of the seven
(*Pyrgo anomala*, *Ammobaculites reophaciformis*) are species-level re-identifications rather than
nomenclatural updates and are flagged there for author confirmation. The identity of the last was
established by specimen number: files named `Ammobaculites reophaciformis_11_*` correspond to
*Nodulina dentaliniformis* S011 in the current manifest.

The same seven names are the reason the audit cohort was mis-scored; see
`processed/manifest/verified/AUDIT_LABEL_TAXONOMY.md`.

### 311 files under current species names

| Cause | Files |
|---|---|
| `gif_rotated90` frames, excluded by `gif_include_rotated90: false` | 168 |
| Specimens absent from the current manifest | 143 |

The rotated-90 frames were generated when that augmentation source was enabled and are now
deliberately excluded by configuration; they remain on disk but out of the dataset.

The remainder belong to specimen identifiers the manifest does not contain:
*Agglutinella soriformis* S021 (125 files), plus `S000` entries under *Karreriella bradyi*,
*Quinqueloculina bosciana*, *Quinqueloculina crassicarinata* and *Triloculina plicata*. `S000` is
not a real specimen number and indicates a specimen-ID parse failure in an early discovery run.
These files are superseded, not lost data: each of those species is represented in the manifest
under its correct specimen number.

## Specimen identifiers are not globally unique

Specimen numbering restarts per species. `S010` denotes one *Pseudotriloculina limbata* and a
different *Pyrgo depressa*, imaged at 0.95 and 1.09 µm per pixel respectively. Any join on
specimen must therefore key on the pair (species, specimen_id). This affects the spatial
calibration table and the dual-stream teacher context, both of which key correctly on the pair.

## Reproducing these numbers

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
v = pd.read_csv("processed/manifest/views_resized.csv")
splits = {k: pd.read_csv(f"processed/manifest/splits/{k}.csv") for k in ("train","val","test")}
print("manifest rows :", len(v))
print("split total   :", sum(len(d) for d in splits.values()))
print("files on disk :", sum(1 for _ in Path("processed/resized").rglob("*.png")))
PY
```
