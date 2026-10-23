#!/usr/bin/env python3
"""Score the frozen validation and test splits with a deployed classifier and write the paper tables.

Per arm this writes, under --out-dir:
  val_predictions.csv / test_predictions.csv      one row per image (top-3, calibrated confidence)
  val_specimens.csv / test_specimens.csv          one row per shell (majority vote and probability sum)
  per_species_test.csv / per_species_val.csv      P / R / F1, shells and images in training, wall type
  confusion_pairs_test.csv                        off-diagonal pairs with genus / family / wall flags
  confusion_matrix_test.csv                       full matrix in display names
  coverage_accuracy_test.csv                      confidence gate curve on the test split
  error_analysis_test.json                        errors by taxonomic distance and wall type
  summary.json                                    headline numbers with Wilson intervals

Test and validation are scored, never used for any decision here; temperature is read from
<checkpoints_dir>/temperature.json, which scripts/calibrate_temperature.py fits on validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classification.transforms import build_eval_transform  # noqa: E402
from src.config.load_config import load_config  # noqa: E402

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
TABLE1 = ROOT / "reports" / "experiments" / "2026-10_drive140_paper" / "table1_species_taxonomy.csv"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def genus_of(name: str) -> str:
    return name.split()[0]


def score_split(model, labels: list[str], input_size: int, csv_path: Path, dev: torch.device, temperature: float) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    tf = build_eval_transform(input_size)
    from PIL import Image

    logits_all = []
    with torch.no_grad():
        batch = []
        for p in df["path_resized"].astype(str):
            batch.append(tf(Image.open(p).convert("RGB")))
            if len(batch) == 32:
                logits_all.append(model(torch.stack(batch).to(dev)).float().cpu())
                batch = []
        if batch:
            logits_all.append(model(torch.stack(batch).to(dev)).float().cpu())
    logits = torch.cat(logits_all).numpy()
    probs_raw = torch.softmax(torch.tensor(logits), dim=1).numpy()
    probs_cal = torch.softmax(torch.tensor(logits) / float(temperature), dim=1).numpy()
    top3 = np.argsort(-probs_cal, axis=1)[:, :3]
    out = df[["species", "specimen_id", "view_id", "source", "path_resized"]].copy()
    out["pred_1"] = [labels[i] for i in top3[:, 0]]
    out["pred_2"] = [labels[i] for i in top3[:, 1]]
    out["pred_3"] = [labels[i] for i in top3[:, 2]]
    out["confidence"] = probs_cal[np.arange(len(out)), top3[:, 0]]
    out["confidence_raw"] = probs_raw[np.arange(len(out)), top3[:, 0]]
    out["correct"] = (out["pred_1"] == out["species"]).astype(int)
    out["top3_correct"] = out[["pred_1", "pred_2", "pred_3"]].eq(out["species"], axis=0).any(axis=1).astype(int)
    out["genus_correct"] = (out["pred_1"].map(genus_of) == out["species"].map(genus_of)).astype(int)
    return out, probs_raw, probs_cal


def specimen_table(pred: pd.DataFrame, probs_cal: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    label_index = {l: i for i, l in enumerate(labels)}
    for (species, sid), part in pred.groupby(["species", "specimen_id"], sort=True):
        votes = Counter(part["pred_1"])
        top_count = max(votes.values())
        leaders = sorted(s for s, c in votes.items() if c == top_count)
        prob_sum = probs_cal[part.index.to_numpy()].sum(axis=0)
        soft = labels[int(prob_sum.argmax())]
        majority = soft if len(leaders) > 1 and soft in leaders else leaders[0]
        rows.append(
            {
                "species": species,
                "specimen_id": int(sid),
                "n_images": int(len(part)),
                "n_correct_images": int(part["correct"].sum()),
                "image_accuracy": float(part["correct"].mean()),
                "majority_prediction": majority,
                "majority_tie": int(len(leaders) > 1),
                "majority_correct": int(majority == species),
                "probsum_prediction": soft,
                "probsum_correct": int(soft == species),
                "mean_confidence": float(part["confidence"].mean()),
                "genus_majority_correct": int(genus_of(majority) == genus_of(species)),
                "true_prob_sum_share": float(prob_sum[label_index[species]] / prob_sum.sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    splits_dir = Path(cfg["paths_resolved"]["splits_dir"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = json.loads((ckpt_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise SystemExit(f"{ckpt_dir} is {status.get('status')!r}; refusing to score a partial run")
    sel = json.loads((ckpt_dir / "selection_provenance.json").read_text(encoding="utf-8"))
    if sel.get("audit_cohort_used") or sel.get("test_split_used"):
        raise SystemExit("selection provenance reports audit or test use; firewall violated")

    ckpt = torch.load(ckpt_dir / "classifier.pt", map_location="cpu")
    labels = json.loads((ckpt_dir / "label_map.json").read_text(encoding="utf-8"))
    input_size = int(ckpt.get("input_size", 384))
    temperature = 1.0
    if (ckpt_dir / "temperature.json").exists():
        temperature = float(json.loads((ckpt_dir / "temperature.json").read_text(encoding="utf-8")).get("temperature", 1.0))

    import timm

    model = timm.create_model(ckpt["backbone"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    dev = device()
    model.to(dev).eval()

    tax = pd.read_csv(TABLE1).set_index("code_label")
    display = tax["display_name"].to_dict()
    wall = tax["wall_type"].to_dict()
    family = tax["family"].to_dict()
    train = pd.read_csv(splits_dir / "train.csv")
    n_train_shells = train.groupby("species")["specimen_id"].nunique().to_dict()
    n_train_images = train.groupby("species").size().to_dict()

    summary: dict[str, object] = {
        "config": args.config,
        "checkpoints_dir": str(ckpt_dir),
        "selected_epoch": status.get("selected_epoch"),
        "epochs_run": status.get("last_epoch"),
        "temperature": temperature,
        "temperature_fit_split": "val",
        "selection": sel,
    }

    for split in ("val", "test"):
        pred, probs_raw, probs_cal = score_split(model, labels, input_size, splits_dir / f"{split}.csv", dev, temperature)
        pred["display_species"] = pred["species"].map(display)
        pred["display_pred_1"] = pred["pred_1"].map(display)
        pred["wall_type"] = pred["species"].map(wall)
        pred.to_csv(out_dir / f"{split}_predictions.csv", index=False)

        shells = specimen_table(pred, probs_cal, labels)
        shells["display_species"] = shells["species"].map(display)
        shells["display_majority"] = shells["majority_prediction"].map(display)
        shells["wall_type"] = shells["species"].map(wall)
        shells["n_train_shells"] = shells["species"].map(n_train_shells)
        shells.to_csv(out_dir / f"{split}_specimens.csv", index=False)

        present = sorted(pred["species"].unique())
        rep = classification_report(pred["species"], pred["pred_1"], labels=present, output_dict=True, zero_division=0)
        per = []
        for s in present:
            m = rep[s]
            sh = shells[shells["species"] == s]
            per.append(
                {
                    "code_label": s,
                    "display_name": display[s],
                    "wall_type": wall[s],
                    "family": family[s],
                    "n_train_shells": int(n_train_shells.get(s, 0)),
                    "n_train_images": int(n_train_images.get(s, 0)),
                    f"n_{split}_images": int(m["support"]),
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1-score"],
                    f"{split}_shell_majority_correct": int(sh["majority_correct"].sum()),
                    f"{split}_shells": int(len(sh)),
                    "image_accuracy": float(pred.loc[pred["species"] == s, "correct"].mean()),
                    "genus_accuracy": float(pred.loc[pred["species"] == s, "genus_correct"].mean()),
                }
            )
        per_df = pd.DataFrame(per).sort_values(["n_train_shells", "f1", "display_name"])
        per_df.to_csv(out_dir / f"per_species_{split}.csv", index=False)

        n = len(pred)
        k = int(pred["correct"].sum())
        ks = int(shells["majority_correct"].sum())
        acc, lo, hi = wilson(k, n)
        sacc, slo, shi = wilson(ks, len(shells))
        correct = pred["correct"].to_numpy()
        block = {
            "n_images": n,
            "n_shells": int(len(shells)),
            "n_species": len(present),
            "image_top1": acc,
            "image_top1_wilson": [lo, hi],
            "image_top1_correct": k,
            "image_top3": float(pred["top3_correct"].mean()),
            "image_genus_top1": float(pred["genus_correct"].mean()),
            "shell_majority": sacc,
            "shell_majority_wilson": [slo, shi],
            "shell_majority_correct": ks,
            "shell_probsum": float(shells["probsum_correct"].mean()),
            "shell_genus_majority": float(shells["genus_majority_correct"].mean()),
            "shell_ties": int(shells["majority_tie"].sum()),
            "by_source": pred.groupby("source")["correct"].agg(["mean", "size"]).rename(columns={"mean": "top1", "size": "n"}).to_dict("index"),
            "by_wall_type": pred.groupby("wall_type")["correct"].agg(["mean", "size"]).rename(columns={"mean": "top1", "size": "n"}).to_dict("index"),
            "shell_majority_by_wall_type": shells.groupby("wall_type")["majority_correct"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"}).to_dict("index"),
            "macro_f1": rep["macro avg"]["f1-score"],
            "ece_raw": ece(probs_raw, correct),
            "ece_calibrated": ece(probs_cal, correct),
            "nll_raw": float(-np.log(probs_raw[np.arange(n), [labels.index(s) for s in pred["species"]]] + 1e-12).mean()),
            "nll_calibrated": float(-np.log(probs_cal[np.arange(n), [labels.index(s) for s in pred["species"]]] + 1e-12).mean()),
            "spearman_f1_vs_train_shells": float(per_df["f1"].corr(per_df["n_train_shells"], method="spearman")) if len(per_df) > 2 else None,
            "spearman_f1_vs_train_images": float(per_df["f1"].corr(per_df["n_train_images"], method="spearman")) if len(per_df) > 2 else None,
        }
        summary[split] = block

        if split == "test":
            cm = confusion_matrix(pred["species"], pred["pred_1"], labels=labels)
            pd.DataFrame(cm, index=[display[l] for l in labels], columns=[display[l] for l in labels]).to_csv(out_dir / "confusion_matrix_test.csv")
            errs = pred[pred["correct"] == 0]
            pairs = (
                errs.groupby(["species", "pred_1"]).size().rename("n").reset_index().sort_values("n", ascending=False)
            )
            pairs["true_display"] = pairs["species"].map(display)
            pairs["pred_display"] = pairs["pred_1"].map(display)
            pairs["same_genus"] = (pairs["species"].map(genus_of) == pairs["pred_1"].map(genus_of)).astype(int)
            pairs["same_family"] = (pairs["species"].map(family) == pairs["pred_1"].map(family)).astype(int)
            pairs["same_wall_type"] = (pairs["species"].map(wall) == pairs["pred_1"].map(wall)).astype(int)
            pairs["true_wall_type"] = pairs["species"].map(wall)
            pairs["pred_wall_type"] = pairs["pred_1"].map(wall)
            pairs.to_csv(out_dir / "confusion_pairs_test.csv", index=False)
            n_err = int(len(errs))
            ea = {
                "n_errors": n_err,
                "same_genus": int(((errs["species"].map(genus_of)) == errs["pred_1"].map(genus_of)).sum()),
                "same_family_different_genus": int(((errs["species"].map(family) == errs["pred_1"].map(family)) & (errs["species"].map(genus_of) != errs["pred_1"].map(genus_of))).sum()),
                "same_wall_type_different_family": int(((errs["species"].map(wall) == errs["pred_1"].map(wall)) & (errs["species"].map(family) != errs["pred_1"].map(family))).sum()),
                "cross_wall_type": int((errs["species"].map(wall) != errs["pred_1"].map(wall)).sum()),
                "errors_by_true_wall_type": errs.groupby("wall_type").size().to_dict(),
                "wall_type_transition_matrix": pd.crosstab(errs["species"].map(wall), errs["pred_1"].map(wall)).to_dict(),
                "missed_shells": shells.loc[shells["majority_correct"] == 0, ["display_species", "specimen_id", "display_majority", "n_correct_images", "n_images", "wall_type"]].to_dict("records"),
            }
            (out_dir / "error_analysis_test.json").write_text(json.dumps(ea, indent=2, default=str), encoding="utf-8")

            curve = []
            for t in THRESHOLDS:
                cov = pred[pred["confidence"] >= t]
                curve.append({"threshold": t, "coverage": len(cov) / n, "n_covered": len(cov), "accuracy_on_covered": float(cov["correct"].mean()) if len(cov) else float("nan")})
            pd.DataFrame(curve).to_csv(out_dir / "coverage_accuracy_test.csv", index=False)
            summary["test_gate_0.9"] = next(c for c in curve if c["threshold"] == 0.9)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("selected_epoch", "temperature")} | {
        "val_image_top1": summary["val"]["image_top1"],
        "val_shell_majority": summary["val"]["shell_majority"],
        "test_image_top1": summary["test"]["image_top1"],
        "test_image_genus_top1": summary["test"]["image_genus_top1"],
        "test_shell_majority": summary["test"]["shell_majority"],
        "test_ece_raw": summary["test"]["ece_raw"],
        "test_ece_calibrated": summary["test"]["ece_calibrated"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
