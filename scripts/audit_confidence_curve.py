"""Confidence and coverage-accuracy statistics for the locked Leica audit cohort.

Derived from the same frozen checkpoint as `eval_audit_once.py`; this script adds no
selection step and must not be used to rank checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.eval_verified_set import _autocrop_otsu_bbox, _device, _letterbox  # noqa: E402
from src.config.load_config import load_config  # noqa: E402
from src.inference.pipeline import load_classifier  # noqa: E402

GATE_ACCEPT = 0.9
GATE_UNKNOWN = 0.2
THRESHOLDS = [0.0, 0.2, 0.35, 0.5, 0.7, 0.9]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels-csv", default="processed/manifest/verified/verified_labels_audit_v2.csv")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    processed = Path(cfg["paths_resolved"]["processed_dir"])

    labels_csv = Path(args.labels_csv)
    if not labels_csv.is_absolute():
        labels_csv = (root / labels_csv).resolve()

    resize_cfg = cfg.get("resize", {}) if isinstance(cfg.get("resize", {}), dict) else {}
    autocrop = str(resize_cfg.get("autocrop", "none")).lower().strip()

    device = _device()
    model, labels, input_size = load_classifier(ckpt_dir)
    model.to(device)
    model.eval()
    temperature = float(getattr(model, "temperature", 1.0))

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    def resolve_image(value: str) -> Path:
        # The verification repository ships the cohort with repo-relative paths; the working
        # manifests hold absolute ones.
        p = Path(str(value))
        return p if p.is_absolute() else (root / p)

    rows = []
    for _, r in pd.read_csv(labels_csv).iterrows():
        im = Image.open(resolve_image(r["image_path"]))
        if autocrop == "otsu_bbox":
            im = _autocrop_otsu_bbox(im)
        x = tf(_letterbox(im, input_size)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
        rows.append(
            {
                "image_path": str(r["image_path"]),
                "true_species": str(r["true_species"]),
                "difficulty_stratum": str(r.get("difficulty_stratum", "")),
                "pred_1": labels[int(idx.item())],
                "confidence": float(conf.item()),
            }
        )

    df = pd.DataFrame(rows)
    df["correct"] = (df["pred_1"] == df["true_species"]).astype(int)

    curve = []
    for t in THRESHOLDS:
        covered = df[df["confidence"] >= t]
        curve.append(
            {
                "threshold": t,
                "coverage": float(len(covered) / len(df)),
                "n_covered": int(len(covered)),
                "accuracy_on_covered": float(covered["correct"].mean()) if len(covered) else float("nan"),
            }
        )

    summary = {
        "config": args.config,
        "checkpoints_dir": str(ckpt_dir),
        "n": int(len(df)),
        "temperature": temperature,
        "audit_cohort_used_for_selection": False,
        "mean_confidence": float(df["confidence"].mean()),
        "mean_confidence_correct": float(df.loc[df["correct"] == 1, "confidence"].mean()),
        "mean_confidence_incorrect": float(df.loc[df["correct"] == 0, "confidence"].mean()),
        "gating": {
            "accept": int((df["confidence"] >= GATE_ACCEPT).sum()),
            "suggest": int(((df["confidence"] >= GATE_UNKNOWN) & (df["confidence"] < GATE_ACCEPT)).sum()),
            "unknown": int((df["confidence"] < GATE_UNKNOWN).sum()),
        },
        "coverage_accuracy_curve": curve,
    }

    prefix = Path(args.out_prefix) if args.out_prefix else processed / "manifest" / "verified" / f"audit_confidence_{ckpt_dir.name}"
    if not prefix.is_absolute():
        prefix = (root / prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(str(prefix) + "_predictions.csv"), index=False)
    Path(str(prefix) + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
