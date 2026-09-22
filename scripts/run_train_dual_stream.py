from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classification.train_dispatch import train_from_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convenience entrypoint for the dual-stream arm. Equivalent to run_train.py; "
            "the arm is selected by the config, not by which script is invoked."
        )
    )
    ap.add_argument("--config", default="configs/train_dual_stream.yaml")
    args = ap.parse_args()

    out = train_from_config(args.config)
    print(f"checkpoint: {out.checkpoint_path}")
    print(f"labels: {out.label_map_path}")


if __name__ == "__main__":
    main()
