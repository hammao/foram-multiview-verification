#!/usr/bin/env python3
"""Phase 0 for the 50-species revision: freeze the evaluation firewall.

Reads the Drive140 view manifest and writes, into a NEW split directory:

  train.csv / val.csv / test.csv       specimen-level split, rule A2
  split_provenance.json                rule, counts, per-species assignment, file hashes
  test_specimens.json                  the frozen test shells (never used for selection)
  audit_overlap_39.csv                 Leica audit images whose species are in the catalogue,
                                       with SHA-256, true species and the Drive140 tier of
                                       every shell of that species
  audit_independence.json              SHA-256 comparison of audit files against every
                                       Drive140 source (TIFF/GIF) and resized PNG

Rule A2 (by number of complete shells per species, ordered by specimen id):
  N >= 4   highest -> test, second-highest -> val, rest -> train
  N == 3   highest -> test, both others -> train
  N == 2   highest -> val, other -> train
  N == 1   train only

The directory is frozen: the script refuses to run if it already exists unless
--force is given. The exploratory split under splits_specimen/ is not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "processed" / "drive140" / "manifest"
VIEWS = MANIFEST / "views_resized.csv"
OUT = MANIFEST / "splits_specimen_tvt"
SPECIES_CSV = ROOT / "configs" / "species_drive140.csv"
AUDIT_CSV = ROOT / "processed" / "manifest" / "verified" / "verified_labels_audit_v2.csv"
SEED = 1337
SF_DATALESS = 0x40000000
COLUMNS = ["species", "specimen_id", "view_id", "path", "source", "z_index", "image_id", "path_resized"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_dataless(path: Path) -> bool:
    try:
        return bool(path.stat().st_flags & SF_DATALESS)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def assign(sids: list[int]) -> dict[int, str]:
    sids = sorted(sids)
    n = len(sids)
    out = {sid: "train" for sid in sids}
    if n >= 4:
        out[sids[-1]] = "test"
        out[sids[-2]] = "val"
    elif n == 3:
        out[sids[-1]] = "test"
    elif n == 2:
        out[sids[-1]] = "val"
    return out


def rule_name(n: int) -> str:
    if n >= 4:
        return "test_highest_val_second_highest"
    if n == 3:
        return "test_highest_two_train"
    if n == 2:
        return "val_highest_no_test"
    return "train_only_singleton"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen split directory")
    ap.add_argument("--skip-independence", action="store_true", help="skip hashing Drive sources (debug only)")
    args = ap.parse_args()

    if OUT.exists() and not args.force:
        print(f"ERROR: {OUT} already exists and is frozen. Use --force only if you intend to re-freeze.", file=sys.stderr)
        return 2
    if not VIEWS.is_file():
        print(f"ERROR: missing {VIEWS}", file=sys.stderr)
        return 2

    df = pd.read_csv(VIEWS)
    df["specimen_id"] = df["specimen_id"].astype(int)
    if df["source"].astype(str).eq("zstack_slice").any():
        print("ERROR: z-stack rows present in manifest", file=sys.stderr)
        return 1

    df["split"] = "train"
    assignments: list[dict[str, object]] = []
    for species, part in df.groupby("species", sort=True):
        sids = sorted({int(x) for x in part["specimen_id"]})
        amap = assign(sids)
        for sid, split in amap.items():
            df.loc[(df["species"] == species) & (df["specimen_id"] == sid), "split"] = split
        assignments.append(
            {
                "species": str(species),
                "n_complete_specimens": len(sids),
                "rule": rule_name(len(sids)),
                "train_specimen_ids": [s for s, v in amap.items() if v == "train"],
                "val_specimen_id": next((s for s, v in amap.items() if v == "val"), None),
                "test_specimen_id": next((s for s, v in amap.items() if v == "test"), None),
            }
        )

    # Invariants
    keys = {s: set(zip(df.loc[df.split == s, "species"], df.loc[df.split == s, "specimen_id"], strict=True)) for s in ("train", "val", "test")}
    if keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"]:
        print("ERROR: a specimen appears in two tiers", file=sys.stderr)
        return 1
    train_species = set(df.loc[df.split == "train", "species"])
    if train_species != set(df["species"]):
        print("ERROR: some species have no training shell", file=sys.stderr)
        return 1
    for a in assignments:
        if a["test_specimen_id"] is not None and len(a["train_specimen_ids"]) < 2:
            print(f"ERROR: {a['species']} tested with fewer than two training shells", file=sys.stderr)
            return 1

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    file_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        part = df.loc[df.split == split, COLUMNS].sort_values(["species", "specimen_id", "view_id", "source", "image_id"])
        dest = OUT / f"{split}.csv"
        part.to_csv(dest, index=False)
        file_hashes[dest.name] = sha256_file(dest)
        summary[split] = {
            "n_images": int(len(part)),
            "n_species": int(part["species"].nunique()),
            "n_specimens": int(part.groupby(["species", "specimen_id"]).ngroups),
            "sources": {str(k): int(v) for k, v in part["source"].value_counts().items()},
        }

    frozen_at = datetime.now(timezone.utc).isoformat()
    test_specimens = {
        "frozen_at_utc": frozen_at,
        "rule": "A2",
        "purpose": "Held-out individuals. Never used for checkpoint selection, early stopping, temperature, thresholds or hyperparameters. Scored once after deploy.",
        "specimens": [
            {"species": a["species"], "specimen_id": a["test_specimen_id"], "n_images": int(((df.species == a["species"]) & (df.specimen_id == a["test_specimen_id"])).sum())}
            for a in assignments
            if a["test_specimen_id"] is not None
        ],
    }
    (OUT / "test_specimens.json").write_text(json.dumps(test_specimens, indent=2), encoding="utf-8")
    file_hashes["test_specimens.json"] = sha256_file(OUT / "test_specimens.json")

    # ---- Audit pre-registration -------------------------------------------------
    catalogue = set(pd.read_csv(SPECIES_CSV)["species"])
    audit = pd.read_csv(AUDIT_CSV)
    audit = audit.loc[audit["include_for_reporting"].astype(str).str.lower().eq("true")].copy()
    n_locked = int(len(audit))
    overlap = audit.loc[audit["true_species"].isin(catalogue)].copy()
    dropped = audit.loc[~audit["true_species"].isin(catalogue), "true_species"].value_counts().to_dict()

    tier_by_species: dict[str, dict[str, list[int]]] = {}
    for a in assignments:
        tier_by_species[a["species"]] = {
            "train": list(a["train_specimen_ids"]),
            "val": [a["val_specimen_id"]] if a["val_specimen_id"] is not None else [],
            "test": [a["test_specimen_id"]] if a["test_specimen_id"] is not None else [],
        }

    rows = []
    audit_hashes: dict[str, str] = {}
    for r in overlap.itertuples(index=False):
        p = Path(str(r.reporting_image_path_abs))
        digest = sha256_file(p)
        audit_hashes[str(p)] = digest
        tiers = tier_by_species[str(r.true_species)]
        rows.append(
            {
                "audit_image_path": str(p),
                "audit_image_basename": p.name,
                "sha256": digest,
                "true_species": str(r.true_species),
                "difficulty_stratum": str(r.difficulty_stratum),
                "cohort_version": str(r.cohort_version),
                "acquisition": "Leica, orientation outside the six training rotations",
                "individuals_relation": "same physical individuals as Drive140 shells of this species, re-imaged",
                "drive140_train_specimen_ids": ";".join(str(s) for s in tiers["train"]),
                "drive140_val_specimen_ids": ";".join(str(s) for s in tiers["val"]),
                "drive140_test_specimen_ids": ";".join(str(s) for s in tiers["test"]),
            }
        )
    overlap_df = pd.DataFrame(rows).sort_values(["true_species", "audit_image_basename"])
    overlap_path = OUT / f"audit_overlap_{len(overlap_df)}.csv"
    overlap_df.to_csv(overlap_path, index=False)
    file_hashes[overlap_path.name] = sha256_file(overlap_path)

    # ---- Independence check -----------------------------------------------------
    independence: dict[str, object] = {"checked_at_utc": frozen_at, "n_audit_files": len(audit_hashes)}
    if not args.skip_independence:
        source_paths = sorted({str(p) for p in df["path"].astype(str)})
        resized_paths = sorted({str(p) for p in df["path_resized"].astype(str)})
        source_hashes: dict[str, str] = {}
        dataless: list[str] = []
        for sp in source_paths:
            pp = Path(sp)
            if is_dataless(pp):
                dataless.append(sp)
                continue
            source_hashes[sp] = sha256_file(pp)
        resized_hashes = {rp: sha256_file(Path(rp)) for rp in resized_paths}
        audit_set = set(audit_hashes.values())
        src_matches = [p for p, h in source_hashes.items() if h in audit_set]
        res_matches = [p for p, h in resized_hashes.items() if h in audit_set]
        audit_basenames = {Path(p).name for p in audit_hashes}
        basename_matches = [p for p in source_paths if Path(p).name in audit_basenames]
        independence.update(
            {
                "n_drive140_source_files_hashed": len(source_hashes),
                "n_drive140_source_files_dataless_skipped": len(dataless),
                "dataless_skipped": dataless,
                "n_drive140_resized_files_hashed": len(resized_hashes),
                "sha256_matches_against_sources": src_matches,
                "sha256_matches_against_resized": res_matches,
                "basename_matches_against_sources": basename_matches,
                "independent": not src_matches and not res_matches and not basename_matches,
                "audit_root": str(Path(next(iter(audit_hashes))).parents[1]) if audit_hashes else None,
                "drive140_root": str(Path(source_paths[0]).parents[4]) if source_paths else None,
            }
        )
    else:
        independence["skipped"] = True
    (OUT / "audit_independence.json").write_text(json.dumps(independence, indent=2), encoding="utf-8")
    file_hashes["audit_independence.json"] = sha256_file(OUT / "audit_independence.json")

    provenance = {
        "frozen_at_utc": frozen_at,
        "seed": SEED,
        "rule": "A2",
        "rule_text": {
            "N>=4": "highest specimen_id -> test; second-highest -> val; remainder -> train",
            "N==3": "highest specimen_id -> test; both others -> train",
            "N==2": "highest specimen_id -> val; other -> train",
            "N==1": "train only",
        },
        "priorities": [
            "a species is tested only if at least two other individuals remain in training",
            "validation needs only a stable selection signal, so it takes the spare shell at N==2 and N>=4",
            "singletons train only",
        ],
        "rejected_alternative": "val shell also taken from 3-shell species: would leave 43 species training on a single individual",
        "manifest": str(VIEWS),
        "manifest_sha256": sha256_file(VIEWS),
        "n_species": int(df["species"].nunique()),
        "n_specimens": int(df.groupby(["species", "specimen_id"]).ngroups),
        "n_images": int(len(df)),
        "multiplicity_profile": {str(k): int(v) for k, v in pd.Series([a["n_complete_specimens"] for a in assignments]).value_counts().sort_index().items()},
        "splits": summary,
        "audit": {
            "locked_cohort_csv": str(AUDIT_CSV),
            "locked_cohort_sha256": sha256_file(AUDIT_CSV),
            "n_locked_reporting_images": n_locked,
            "n_in_catalogue": int(len(overlap_df)),
            "dropped_out_of_catalogue": {str(k): int(v) for k, v in dropped.items()},
            "overlap_file": overlap_path.name,
            "usage": "final evaluation only; scored once via scripts/eval_audit_once.py",
        },
        "file_sha256": file_hashes,
        "assignments": assignments,
    }
    (OUT / "split_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Freeze: read-only files
    for p in OUT.iterdir():
        os.chmod(p, 0o444)

    print(json.dumps({"out": str(OUT), "splits": summary, "audit_overlap": int(len(overlap_df)), "independent": independence.get("independent")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
