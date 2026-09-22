from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.config.load_config import load_config
from src.inference.preprocess import _autocrop_otsu_bbox, _letterbox


class _TemperatureScaledModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, temperature: float):
        super().__init__()
        self.base = base
        self.temperature = float(temperature)

    def forward(self, x):  # noqa: ANN001
        logits = self.base(x)
        T = float(self.temperature) if self.temperature and self.temperature > 0 else 1.0
        return logits / T


def load_classifier(checkpoints_dir: Path):
    ckpt_path = checkpoints_dir / "classifier.pt"
    labels_path = checkpoints_dir / "label_map.json"
    temp_path = checkpoints_dir / "temperature.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {ckpt_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing label map: {labels_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt["backbone"]
    input_size = int(ckpt.get("input_size", 384))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    temperature = 1.0
    if temp_path.exists():
        try:
            temperature = float(json.loads(temp_path.read_text(encoding="utf-8")).get("temperature", 1.0))
        except Exception:
            temperature = 1.0

    import timm

    model = timm.create_model(backbone, pretrained=False, num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    if temperature != 1.0:
        model = _TemperatureScaledModel(model, temperature=temperature)
    model.eval()
    return model, labels, input_size


def predict_image(
    image_path: Path,
    config_path: str = "configs/default.yaml",
    *,
    top_k: int | None = None,
    accept_threshold: float | None = None,
    unknown_threshold: float | None = None,
) -> dict:
    cfg = load_config(config_path)
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])

    inf_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference", {}), dict) else {}
    top_k = int(top_k if top_k is not None else inf_cfg.get("top_k", 3))
    accept_threshold = float(
        accept_threshold if accept_threshold is not None else inf_cfg.get("accept_threshold", 0.9)
    )
    unknown_threshold = float(
        unknown_threshold if unknown_threshold is not None else inf_cfg.get("unknown_threshold", 0.6)
    )
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not (0.0 <= unknown_threshold <= accept_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy 0 <= unknown_threshold <= accept_threshold <= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, input_size = load_classifier(checkpoints_dir)
    model.to(device)

    im = Image.open(image_path)
    resize_cfg = cfg.get("resize", {}) if isinstance(cfg.get("resize", {}), dict) else {}
    if str(resize_cfg.get("autocrop", "none")).lower().strip() == "otsu_bbox":
        im = _autocrop_otsu_bbox(
            im,
            margin_frac=float(resize_cfg.get("autocrop_margin_frac", 0.06)),
            min_area_frac=float(resize_cfg.get("autocrop_min_area_frac", 0.01)),
        )
    im = _letterbox(im, input_size)

    # Basic normalization compatible with timm defaults (ImageNet)
    import torchvision.transforms as T

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    x = tf(im).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probs.numel())
        top_probs, top_idxs = torch.topk(probs, k=k, largest=True, sorted=True)

    suggestions = [
        {"species": labels[int(i.item())], "confidence": float(p.item())}
        for p, i in zip(top_probs, top_idxs)
    ]
    best = suggestions[0]
    best_conf = float(best["confidence"])
    entropy = float(-(probs * torch.log(torch.clamp(probs, min=1e-12))).sum().item())
    if probs.numel() >= 2:
        margin = float((top_probs[0] - top_probs[1]).item())
    else:
        margin = float("nan")
    # Energy score (lower is typically more in-distribution). Uses current logits (optionally temperature-scaled).
    energy = float((-torch.logsumexp(logits[0], dim=0)).item())

    if best_conf >= accept_threshold:
        decision = "accept"
        predicted_species = str(best["species"])
    elif best_conf >= unknown_threshold:
        decision = "suggest"
        predicted_species = str(best["species"])
    else:
        decision = "unknown"
        predicted_species = None

    return {
        "decision": decision,  # accept | suggest | unknown
        "predicted_species": predicted_species,
        "confidence": best_conf,
        "entropy": entropy,
        "margin": margin,
        "energy": energy,
        "suggestions": suggestions,
        "calibration": {"temperature": float(getattr(model, "temperature", 1.0))},
        "thresholds": {
            "accept_threshold": accept_threshold,
            "unknown_threshold": unknown_threshold,
            "top_k": top_k,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--accept-threshold", type=float, default=None)
    ap.add_argument("--unknown-threshold", type=float, default=None)
    args = ap.parse_args()

    out = predict_image(
        Path(args.image),
        config_path=args.config,
        top_k=args.top_k,
        accept_threshold=args.accept_threshold,
        unknown_threshold=args.unknown_threshold,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

