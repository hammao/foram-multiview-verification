#!/usr/bin/env python3
"""Recompute the quantitative claims of the revised manuscript from the released files."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = json.loads((ROOT / "expected_values.json").read_text())
rows = []


def check(group, label, actual, expected, places=3):
    if isinstance(expected, float):
        ok = round(float(actual), places) == round(float(expected), places)
        a, e = f"{float(actual):.{places}f}", f"{float(expected):.{places}f}"
    else:
        ok = actual == expected
        a, e = str(actual), str(expected)
    rows.append((group, label, ok, a, e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group")
    args = ap.parse_args()
    d = EXP["dataset"]
    S = ROOT / "data" / "splits"
    frames = {s: pd.read_csv(S / f"{s}.csv") for s in ("train", "val", "test")}
    g = "partition"
    for s, f in frames.items():
        check(g, f"{s} images", len(f), d["splits"][s]["n_images"])
        check(g, f"{s} shells", f.groupby(["species", "specimen_id"]).ngroups, d["splits"][s]["n_specimens"])
        check(g, f"{s} species", f.species.nunique(), d["splits"][s]["n_species"])
    allv = pd.concat(frames.values())
    check(g, "species", allv.species.nunique(), d["n_species"])
    check(g, "specimens", allv.groupby(["species", "specimen_id"]).ngroups, d["n_specimens"])
    check(g, "images", len(allv), d["n_images"])
    sh = lambda f: set(zip(f.species, f.specimen_id))  # noqa: E731
    check(g, "specimens shared between tiers", len((sh(frames["train"]) & sh(frames["val"])) | (sh(frames["train"]) & sh(frames["test"])) | (sh(frames["val"]) & sh(frames["test"]))), 0)
    ntrain = frames["train"].groupby("species").specimen_id.nunique()
    check(g, "test species with fewer than two training specimens", int((ntrain.reindex(frames["test"].species.unique()) < 2).sum()), 0)
    for name, h in d["file_sha256"].items():
        p = S / name
        if p.exists():
            check(g, f"sha256 {name}", hashlib.sha256(p.read_bytes()).hexdigest(), h)
    ind = json.loads((S / "audit_independence.json").read_text())
    n_matches = sum(len(ind.get(k, [])) if isinstance(ind.get(k), list) else int(ind.get(k, 0) or 0) for k in ("sha256_matches_against_sources", "sha256_matches_against_resized", "basename_matches_against_sources"))
    check(g, "audit files matching catalogue files", n_matches, 0)
    check(g, "audit independent", bool(ind.get("independent")), True)
    check(g, "audit cohort size", len(pd.read_csv(S / "audit_overlap_39.csv")), 39)

    for arm, e in EXP.items():
        if arm == "dataset":
            continue
        g = arm
        R = ROOT / "results" / arm
        t = pd.read_csv(R / "test_predictions.csv")
        check(g, "test image top-1", t.correct.mean(), e["test_image_top1"])
        check(g, "test correct images", int(t.correct.sum()), e["test_image_top1_correct"])
        check(g, "test genus top-1", t.genus_correct.mean(), e["test_image_genus_top1"])
        check(g, "test top-3", t.top3_correct.mean(), e["test_image_top3"])
        sp = pd.read_csv(R / "test_specimens.csv")
        check(g, "test specimen majority", sp.majority_correct.mean(), e["test_shell_majority"])
        check(g, "test specimens correct", int(sp.majority_correct.sum()), e["test_shell_majority_correct"])
        v = pd.read_csv(R / "val_predictions.csv")
        check(g, "val image top-1", v.correct.mean(), e["val_image_top1"])
        prov = json.loads((R / "selection_provenance.json").read_text())
        check(g, "audit cohort used for selection", bool(prov.get("audit_cohort_used", True)), False)
        check(g, "test split used for selection", bool(prov.get("test_split_used", True)), False)
        st = json.loads((R / "run_status.json").read_text())
        check(g, "run complete", st.get("status"), "complete")
        check(g, "selected epoch", st.get("selected_epoch"), e["selected_epoch"])
        if (R / "audit_once_39_predictions.csv").exists() and "audit_top1" in e:
            a = pd.read_csv(R / "audit_once_39_predictions.csv")
            check(g, "audit n", len(a), 39)
            check(g, "audit top-1", (a.pred_1 == a.true_species).mean(), e["audit_top1"])
            top3 = ((a.pred_1 == a.true_species) | (a.pred_2 == a.true_species) | (a.pred_3 == a.true_species)).mean()
            check(g, "audit top-3", top3, e["audit_top3"])

    sel = [r for r in rows if args.group is None or r[0] == args.group]
    cur = None
    for grp, label, ok, a, e in sel:
        if grp != cur:
            print(f"\n{grp}")
            cur = grp
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<50} {a if ok else a + '  (paper says ' + e + ')'}")
    failed = [r for r in sel if not r[2]]
    print(f"\n{len(sel) - len(failed)}/{len(sel)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
