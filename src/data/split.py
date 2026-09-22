from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def stable_hash(text: str) -> int:
    """
    Process-independent replacement for the builtin hash() when deriving RNG seeds.

    Python salts str hashing per process unless PYTHONHASHSEED is set, so seeding an
    RNG from hash(species) produced a different split on every run and made the
    published train/val/test manifests impossible to regenerate from the seed alone.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    allow_view_split_when_insufficient: bool = True
    # Strategy:
    # - "specimen": prefer specimen-level split within each species (no leakage).
    # - "holdout_views": hold out entire view_id(s) within each (species, specimen_id) so val covers
    #   many species even when there is only one specimen per species. This is a *within-specimen*
    #   generalization test (angle/view holdout), not a specimen holdout.
    strategy: str = "specimen"
    # For holdout_views: cap the number of z-stack slices per (species, specimen_id, view_id)
    # in validation/test to avoid overwhelming splits with many near-duplicate slices.
    # If None: keep all slices.
    max_z_slices_per_view_in_val: int | None = None
    max_z_slices_per_view_in_test: int | None = None


def _check_ratios(cfg: SplitConfig) -> None:
    s = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0; got {s}")


def split_manifest(manifest_csv: Path, out_dir: Path, cfg: SplitConfig) -> dict[str, Path]:
    _check_ratios(cfg)
    df = pd.read_csv(manifest_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        paths = {}
        for split in ["train", "val", "test"]:
            p = out_dir / f"{split}.csv"
            df.to_csv(p, index=False)
            paths[split] = p
        return paths

    # Prefer specimen-level split *within each species*.
    #
    # Important: a global specimen split can accidentally place entire species into test only
    # when there is ~1 specimen per species (train∩test becomes empty), yielding misleading 0% accuracy.
    rng = np.random.default_rng(cfg.seed)
    df = df.copy()
    df["specimen_key"] = df["species"].astype(str) + "::" + df["specimen_id"].astype(int).astype(str)
    df.loc[:, "split"] = "train"
    strategy = str(getattr(cfg, "strategy", "specimen")).lower().strip()

    def _subsample_z_slices(
        df_in: pd.DataFrame,
        *,
        split_name: str,
        max_per_view: int | None,
        seed_offset: int,
    ) -> pd.DataFrame:
        if max_per_view is None:
            return df_in
        if max_per_view <= 0:
            return df_in
        if "source" not in df_in.columns:
            return df_in
        if not {"species", "specimen_id", "view_id"}.issubset(set(df_in.columns)):
            return df_in

        z = df_in[(df_in["split"] == split_name) & (df_in["source"].astype(str) == "zstack_slice")].copy()
        if z.empty:
            return df_in
        keep_idx: list[int] = []
        drop_idx: list[int] = []
        for (species, specimen_id, view_id), df_g in z.groupby(["species", "specimen_id", "view_id"], sort=False):
            idx = df_g.index.to_numpy()
            # Deterministic per-group seed so repeated runs are stable.
            rng_g = np.random.default_rng(
                int(cfg.seed)
                + int(seed_offset)
                + (stable_hash(str(species)) % 10_000)
                + int(specimen_id) * 100
                + int(view_id) * 10
            )
            rng_g.shuffle(idx)
            k = min(int(max_per_view), int(len(idx)))
            keep_idx.extend(idx[:k].tolist())
            drop_idx.extend(idx[k:].tolist())

        df_out = df_in.copy()
        if drop_idx:
            df_out = df_out.drop(index=drop_idx)
        return df_out

    def _alloc_counts(n: int) -> tuple[int, int, int]:
        """Return (n_train, n_val, n_test) with sane minima and sums."""
        if n <= 1:
            return (n, 0, 0)
        n_val = int(n * cfg.val_ratio)
        n_test = int(n * cfg.test_ratio)
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            # take from val/test if needed
            if n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
        # Ensure non-empty test/val when ratios request it and n is large enough.
        if cfg.test_ratio > 0 and n >= 2 and n_test == 0:
            n_test = 1
            n_train = max(1, n - n_val - n_test)
        # Validation:
        # - If you're requesting BOTH val and test, you generally need >=3 specimens to avoid collapsing train.
        # - If you're requesting val only (test_ratio==0), allow val for n>=2 by holding out 1 specimen.
        if cfg.val_ratio > 0 and n_val == 0:
            if cfg.test_ratio <= 0 and n >= 2:
                n_val = 1
                n_train = max(1, n - n_val - n_test)
            elif n >= 3:
                n_val = 1
                n_train = max(1, n - n_val - n_test)
        # Final guard: preserve sum
        n_train = max(1, n - n_val - n_test)
        n_val = max(0, min(n_val, n - n_train))
        n_test = max(0, n - n_train - n_val)
        return (n_train, n_val, n_test)

    if strategy == "holdout_views":
        # Hold out entire view_id(s) within each (species, specimen_id).
        if "view_id" not in df.columns:
            raise ValueError("split strategy holdout_views requires 'view_id' column in manifest.")

        for (species, specimen_id), df_g in df.groupby(["species", "specimen_id"], sort=True):
            view_ids = sorted({int(v) for v in df_g["view_id"].dropna().astype(int).tolist()})
            if not view_ids:
                continue

            # Determine number of views to hold out for val/test.
            n_views = len(view_ids)
            n_val = int(round(n_views * float(cfg.val_ratio)))
            n_test = int(round(n_views * float(cfg.test_ratio)))
            if float(cfg.val_ratio) > 0 and n_val < 1:
                n_val = 1
            if float(cfg.test_ratio) > 0 and n_test < 1:
                n_test = 1
            # Keep at least 1 view for train.
            if (n_val + n_test) >= n_views:
                n_val = min(n_val, max(0, n_views - 1))
                n_test = min(n_test, max(0, n_views - 1 - n_val))

            # Deterministic shuffle per group.
            rng_g = np.random.default_rng(
                int(cfg.seed) + (stable_hash(str(species)) % 10_000) + int(specimen_id) * 100
            )
            view_ids_shuf = view_ids[:]
            rng_g.shuffle(view_ids_shuf)
            val_views = set(view_ids_shuf[:n_val])
            test_views = set(view_ids_shuf[n_val : n_val + n_test])

            m_val = (df["species"] == species) & (df["specimen_id"] == specimen_id) & (df["view_id"].astype(int).isin(val_views))
            m_test = (df["species"] == species) & (df["specimen_id"] == specimen_id) & (df["view_id"].astype(int).isin(test_views))
            df.loc[m_val, "split"] = "val"
            df.loc[m_test, "split"] = "test"
            # Remaining views stay train.
    else:
        # Default: specimen-level split within each species, with optional leaky view-row fallback.
        for species, df_s in df.groupby("species", sort=True):
            specimen_keys = df_s["specimen_key"].drop_duplicates().tolist()
            rng_s = np.random.default_rng(cfg.seed + (stable_hash(str(species)) % 10_000))
            rng_s.shuffle(specimen_keys)

            # Specimen-level split whenever possible (>=2 specimens) to avoid leakage
            # (i.e., no views from the same physical specimen should appear in multiple splits).
            if len(specimen_keys) >= 2:
                n_train, n_val, n_test = _alloc_counts(len(specimen_keys))
                train_keys = set(specimen_keys[:n_train])
                val_keys = set(specimen_keys[n_train : n_train + n_val])
                test_keys = set(specimen_keys[n_train + n_val : n_train + n_val + n_test])
                df.loc[df["specimen_key"].isin(test_keys), "split"] = "test"
                df.loc[df["specimen_key"].isin(val_keys), "split"] = "val"
                df.loc[df["specimen_key"].isin(train_keys), "split"] = "train"
            elif cfg.allow_view_split_when_insufficient:
                # Too few specimens for species-level holdout; do a *view-row* split within the species
                # so evaluation artifacts can still be generated for debugging.
                #
                # NOTE: This is inherently leaky when there is only one specimen for the species,
                # because different views of the same specimen can land in both train and test.
                idx = df_s.index.to_numpy()
                rng_s.shuffle(idx)
                n = len(idx)
                n_train = max(1, int(n * cfg.train_ratio))
                n_val = max(0, int(n * cfg.val_ratio))
                train_idx = idx[:n_train]
                val_idx = idx[n_train : n_train + n_val]
                test_idx = idx[n_train + n_val :]
                df.loc[test_idx, "split"] = "test"
                df.loc[val_idx, "split"] = "val"
                df.loc[train_idx, "split"] = "train"
            else:
                # Single-specimen species: everything to train (no evaluation possible for this species).
                df.loc[df_s.index, "split"] = "train"

    # Optional slice subsampling (helps make val/test reflect diversity rather than slice count).
    df = _subsample_z_slices(
        df,
        split_name="val",
        max_per_view=getattr(cfg, "max_z_slices_per_view_in_val", None),
        seed_offset=11,
    )
    df = _subsample_z_slices(
        df,
        split_name="test",
        max_per_view=getattr(cfg, "max_z_slices_per_view_in_test", None),
        seed_offset=22,
    )

    paths: dict[str, Path] = {}
    for split in ["train", "val", "test"]:
        p = out_dir / f"{split}.csv"
        df[df["split"] == split].drop(columns=["specimen_key"], errors="ignore").to_csv(p, index=False)
        paths[split] = p

    # Also write a single CSV with split column.
    all_p = out_dir / "all_splits.csv"
    df.drop(columns=["specimen_key"], errors="ignore").to_csv(all_p, index=False)
    paths["all"] = all_p

    return paths

