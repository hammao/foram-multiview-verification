from __future__ import annotations

import json
import sys
from pathlib import Path

import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classification.eval import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    out = evaluate(args.config)
    print(json.dumps({"backbone": out["backbone"], "input_size": out["input_size"]}, indent=2))


if __name__ == "__main__":
    main()

