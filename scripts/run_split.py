from __future__ import annotations

import sys
from pathlib import Path

import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.load_config import load_config
from src.data.split import SplitConfig, split_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    resized_manifest = processed_dir / "manifest" / "views_resized.csv"

    split_cfg = cfg.get("split", {})
    out = split_manifest(
        resized_manifest,
        out_dir=processed_dir / "manifest" / "splits",
        cfg=SplitConfig(
            seed=int(split_cfg.get("seed", 1337)),
            train_ratio=float(split_cfg.get("train_ratio", 0.8)),
            val_ratio=float(split_cfg.get("val_ratio", 0.1)),
            test_ratio=float(split_cfg.get("test_ratio", 0.1)),
            allow_view_split_when_insufficient=bool(split_cfg.get("allow_view_split_when_insufficient", True)),
            strategy=str(split_cfg.get("strategy", "specimen")),
            max_z_slices_per_view_in_val=(
                int(split_cfg.get("max_z_slices_per_view_in_val"))
                if split_cfg.get("max_z_slices_per_view_in_val", None) is not None
                else None
            ),
            max_z_slices_per_view_in_test=(
                int(split_cfg.get("max_z_slices_per_view_in_test"))
                if split_cfg.get("max_z_slices_per_view_in_test", None) is not None
                else None
            ),
        ),
    )
    for k, p in out.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()

