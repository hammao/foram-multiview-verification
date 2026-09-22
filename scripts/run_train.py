from __future__ import annotations

import sys
from pathlib import Path

import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classification.train_dispatch import train_from_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train either arm. The arm is selected by classification.use_multi_view_distillation in the config."
    )
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    out = train_from_config(args.config)
    print(f"checkpoint: {out.checkpoint_path}")
    print(f"labels: {out.label_map_path}")


if __name__ == "__main__":
    main()

