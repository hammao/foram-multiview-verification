"""Evaluate the locked 45-image Leica cohort exactly once. Firewall: no selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.eval_verified_set import _device, _predict_topk  # noqa: E402
from src.config.load_config import load_config  # noqa: E402
from src.inference.pipeline import load_classifier  # noqa: E402


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _topk_hits(df: pd.DataFrame, k: int) -> int:
    cols = [f"pred_{i}" for i in range(1, k + 1) if f"pred_{i}" in df.columns]
    return int(df[cols].eq(df["true_species"], axis=0).any(axis=1).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description="Locked audit evaluation. Run once per frozen checkpoint.")
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

    df = pd.read_csv(labels_csv)

    def resolve_image(value: str) -> Path:
        # The verification repository ships the cohort with repo-relative paths; the working
        # manifests hold absolute ones.
        p = Path(str(value))
        return p if p.is_absolute() else (root / p)

    device = _device()
    model, labels, input_size = load_classifier(ckpt_dir)
    model.to(device)
    model.eval()

    resize_cfg = cfg.get("resize", {}) if isinstance(cfg.get("resize", {}), dict) else {}
    autocrop = str(resize_cfg.get("autocrop", "none"))

    rows = []
    for _, r in df.iterrows():
        preds = _predict_topk(
            model=model,
            labels=labels,
            input_size=input_size,
            image_path=resolve_image(r["image_path"]),
            device=device,
            top_k=3,
            autocrop=autocrop,
        )
        row = {
            "image_path": str(r["image_path"]),
            "true_species": str(r["true_species"]),
            "difficulty_stratum": str(r.get("difficulty_stratum", "")),
        }
        for i, s in enumerate(preds, start=1):
            row[f"pred_{i}"] = s
        rows.append(row)
    out = pd.DataFrame(rows)

    def pack(sub: pd.DataFrame, name: str) -> dict:
        n = int(len(sub))
        t1 = _topk_hits(sub, 1)
        t3 = _topk_hits(sub, 3)
        p1, lo1, hi1 = wilson(t1, n)
        p3, lo3, hi3 = wilson(t3, n)
        return {
            "stratum": name,
            "n": n,
            "top1": p1,
            "top1_wilson_lo": lo1,
            "top1_wilson_hi": hi1,
            "top3": p3,
            "top3_wilson_lo": lo3,
            "top3_wilson_hi": hi3,
            "top1_correct": t1,
            "top3_correct": t3,
        }

    summary = {
        "config": args.config,
        "checkpoints_dir": str(ckpt_dir),
        "labels_csv": str(labels_csv),
        "n": int(len(out)),
        "audit_cohort_used_for_selection": False,
        "pooled": pack(out, "pooled"),
        "normal": pack(out[out["difficulty_stratum"].str.lower() == "normal"], "normal"),
        "difficult": pack(out[out["difficulty_stratum"].str.lower() == "difficult"], "difficult"),
    }

    prefix = Path(args.out_prefix) if args.out_prefix else processed / "manifest" / "verified" / f"audit_once_{ckpt_dir.name}"
    if not prefix.is_absolute():
        prefix = (root / prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(str(prefix) + "_predictions.csv"), index=False)
    Path(str(prefix) + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
