from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import strftime

import pandas as pd
import torch
from PIL import Image

# Allow running as a script without installing the package.
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.load_config import load_config
from src.inference.pipeline import load_classifier
from src.inference.preprocess import _autocrop_otsu_bbox, _letterbox


@dataclass(frozen=True)
class EvalResult:
    n: int
    top1: float
    top2: float
    top3: float
    top5: float
    top10: float


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _predict_topk(
    *,
    model: torch.nn.Module,
    labels: list[str],
    input_size: int,
    image_path: Path,
    device: torch.device,
    top_k: int,
    autocrop: str,
) -> list[str]:
    # Basic normalization compatible with timm defaults (ImageNet)
    import torchvision.transforms as T

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    im = Image.open(image_path)
    if str(autocrop).lower().strip() == "otsu_bbox":
        im = _autocrop_otsu_bbox(im)
    im = _letterbox(im, input_size)
    x = tf(im).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(int(top_k), int(probs.numel()))
        _, top_idxs = torch.topk(probs, k=k, largest=True, sorted=True)

    return [labels[int(i.item())] for i in top_idxs]


def _topk_acc(df: pd.DataFrame, *, k: int) -> float:
    cols = [f"pred_{i}" for i in range(1, k + 1)]
    return float(df[cols].eq(df["true_species"], axis=0).any(axis=1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained model on a verified labeled image list.")
    ap.add_argument("--config", default="configs/train_all_genus_autocrop.yaml")
    ap.add_argument("--labels-csv", default="processed/manifest/verified/verified_labels.csv")
    ap.add_argument("--out-json", default=None, help="Optional output JSON (default: processed/manifest/verified/...).")
    ap.add_argument("--out-csv", default=None, help="Optional per-image predictions CSV (default: processed/manifest/verified/...).")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--autocrop",
        default=None,
        choices=["none", "otsu_bbox"],
        help="Override preprocessing autocrop for evaluation (default: use config resize.autocrop).",
    )
    ap.add_argument("--max-images", type=int, default=None, help="Optional cap for quick runs.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])

    labels_csv = Path(args.labels_csv)
    if not labels_csv.is_absolute():
        labels_csv = (project_root / labels_csv).resolve()
    df = pd.read_csv(labels_csv)
    if "image_path" not in df.columns or "true_species" not in df.columns:
        raise SystemExit(f"{labels_csv} must contain columns: image_path,true_species")
    if args.max_images is not None:
        df = df.iloc[: int(args.max_images)].copy()

    resize_cfg = cfg.get("resize", {}) if isinstance(cfg.get("resize", {}), dict) else {}
    autocrop = str(args.autocrop) if args.autocrop is not None else str(resize_cfg.get("autocrop", "none"))

    device = _device()
    model, labels, input_size = load_classifier(checkpoints_dir)
    model.to(device)
    model.eval()

    rows: list[dict] = []
    errors = 0
    for _, r in df.iterrows():
        p = Path(str(r["image_path"]))
        true = str(r["true_species"])
        try:
            preds = _predict_topk(
                model=model,
                labels=labels,
                input_size=input_size,
                image_path=p,
                device=device,
                top_k=int(args.top_k),
                autocrop=autocrop,
            )
            row = {"image_path": str(p.resolve()), "true_species": true}
            for i, s in enumerate(preds, start=1):
                row[f"pred_{i}"] = s
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            errors += 1
            rows.append({"image_path": str(p), "true_species": true, "error": repr(e)})

    out_df = pd.DataFrame(rows)
    # Ensure columns exist up to max-k even if some rows errored.
    for i in range(1, int(args.top_k) + 1):
        c = f"pred_{i}"
        if c not in out_df.columns:
            out_df[c] = None

    metrics = EvalResult(
        n=int(len(out_df)),
        top1=_topk_acc(out_df, k=1),
        top2=_topk_acc(out_df, k=min(2, int(args.top_k))),
        top3=_topk_acc(out_df, k=min(3, int(args.top_k))),
        top5=_topk_acc(out_df, k=min(5, int(args.top_k))),
        top10=_topk_acc(out_df, k=min(10, int(args.top_k))),
    )

    out_dir = processed_dir / "manifest" / "verified"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else (out_dir / f"eval_verified_{strftime('%Y-%m-%d_%H%M%S')}.json")
    out_csv = Path(args.out_csv) if args.out_csv else (out_dir / f"eval_verified_{strftime('%Y-%m-%d_%H%M%S')}.csv")
    if not out_json.is_absolute():
        out_json = (project_root / out_json).resolve()
    if not out_csv.is_absolute():
        out_csv = (project_root / out_csv).resolve()

    payload = {
        "labels_csv": str(labels_csv),
        "config": str(Path(args.config)),
        "device": str(device),
        "autocrop": str(autocrop),
        "top_k": int(args.top_k),
        "n": int(metrics.n),
        "n_errors": int(errors),
        "top1": float(metrics.top1),
        "top2": float(metrics.top2),
        "top3": float(metrics.top3),
        "top5": float(metrics.top5),
        "top10": float(metrics.top10),
        "out_csv": str(out_csv),
    }

    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

