"""Leave-one-specimen-out split for Hoeglundina elegans, the only 2-specimen species."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-splits", default="processed/manifest/splits/all_splits.csv")
    ap.add_argument(
        "--from-modelled-splits",
        action="store_true",
        default=True,
        help="Use the concatenated train/val/test CSVs (5,967 modelled images) instead of uncapped all_splits.",
    )
    ap.add_argument("--out-dir", default="processed/manifest/splits/helegans_loso")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    splits_dir = root / "processed" / "manifest" / "splits"
    if args.from_modelled_splits:
        df = pd.concat(
            [pd.read_csv(splits_dir / name) for name in ("train.csv", "val.csv", "test.csv")],
            ignore_index=True,
        )
    else:
        src = Path(args.all_splits)
        if not src.is_absolute():
            src = root / src
        df = pd.read_csv(src)
    he = df["species"] == "Hoeglundina elegans"
    specimens = sorted(df.loc[he, "specimen_id"].dropna().astype(int).unique().tolist())
    if len(specimens) != 2:
        raise SystemExit(f"expected 2 H. elegans specimens, found {specimens}")

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    for held in specimens:
        fold = out_root / f"hold_out_{held}"
        fold.mkdir(parents=True, exist_ok=True)
        train = df[~(he & (df["specimen_id"].astype(int) == held))].copy()
        test = df[he & (df["specimen_id"].astype(int) == held)].copy()
        train.to_csv(fold / "train.csv", index=False)
        # Reuse the existing val split minus the held specimen so early stopping
        # does not see the LOSO test individual.
        val = df[(df.get("split") == "val") & ~(he & (df["specimen_id"].astype(int) == held))].copy()
        if val.empty:
            val = train.iloc[0:0].copy()
        val.to_csv(fold / "val.csv", index=False)
        test.to_csv(fold / "test.csv", index=False)
        print(f"hold_out_{held}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
