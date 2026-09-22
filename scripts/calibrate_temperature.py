from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classification.dataset import ViewDataset, build_label_space
from src.config.load_config import load_config
from src.data.species import load_species_list


@dataclass(frozen=True)
class CalibResult:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    n_samples: int


def _ece(probs: torch.Tensor, y: torch.Tensor, *, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) on top-1 confidence.
    probs: [N,C] softmax probabilities
    y: [N] int labels
    """
    conf, pred = torch.max(probs, dim=1)
    acc = (pred == y).float()
    bins = torch.linspace(0, 1, steps=n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        m = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        if not torch.any(m):
            continue
        w = m.float().mean()
        ece = ece + w * torch.abs(acc[m].mean() - conf[m].mean())
    return float(ece.detach().cpu().item())


def _fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> CalibResult:
    """
    Fit a single temperature parameter on a fixed (logits, y) set.
    Uses LBFGS on NLL with a softplus parameterization to keep T>0.
    """
    device = logits.device
    y = y.to(device)

    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        nll_before = float(ce(logits, y).detach().cpu().item())
        ece_before = _ece(torch.softmax(logits, dim=1), y)

    t_raw = torch.nn.Parameter(torch.tensor([0.0], device=device))  # softplus(0)=0.693...
    opt = torch.optim.LBFGS([t_raw], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.nn.functional.softplus(t_raw) + 1e-6
        loss = ce(logits / T, y)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        T = float((torch.nn.functional.softplus(t_raw) + 1e-6).item())
        nll_after = float(ce(logits / T, y).detach().cpu().item())
        ece_after = _ece(torch.softmax(logits / T, dim=1), y)

    return CalibResult(
        temperature=T,
        nll_before=nll_before,
        nll_after=nll_after,
        ece_before=ece_before,
        ece_after=ece_after,
        n_samples=int(y.numel()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit temperature scaling on the validation split.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Which split to calibrate on (default: val).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: checkpoints/temperature.json).")
    ap.add_argument("--checkpoints-dir", default=None, help="Override checkpoints dir (default: from config).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    project_root = Path(__file__).resolve().parent.parent
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else Path(cfg["paths_resolved"]["checkpoints_dir"])
    if not checkpoints_dir.is_absolute():
        checkpoints_dir = (project_root / checkpoints_dir).resolve()
    species_csv = Path(cfg["paths_resolved"]["species_csv"])

    split_csv = Path(cfg["paths_resolved"]["splits_dir"]) / f"{args.split}.csv"
    if not split_csv.exists():
        raise SystemExit(f"Missing split CSV: {split_csv}")
    try:
        import pandas as pd

        df_check = pd.read_csv(split_csv)
        if df_check.empty:
            raise SystemExit(
                f"Split '{args.split}' is empty under the current no-leak splitting policy. "
                f"Add more specimens per species (>=3) or calibrate on a different split."
            )
    except Exception:
        # If pandas read fails, let ViewDataset raise a clearer error below.
        pass

    ckpt_path = checkpoints_dir / "classifier.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    # Device: use MPS/CUDA if available to speed up calibration.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    species_list = load_species_list(species_csv)
    label_space = build_label_space(species_list)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt["backbone"]
    input_size = int(ckpt.get("input_size", cfg.get("resize", {}).get("input_size", 384)))

    import timm

    model = timm.create_model(backbone, pretrained=False, num_classes=len(species_list))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = None
    try:
        tf = timm.data.create_transform(input_size=(3, input_size, input_size), is_training=False)
    except Exception:
        tf = None

    ds = ViewDataset(split_csv, label_space=label_space, transform=tf, root=cfg["paths_resolved"].get("base_dir"))
    num_workers = 2 if device.type == "cuda" else 0
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=num_workers)

    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb).detach().float().cpu()
            all_logits.append(logits)
            all_y.append(yb.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    if logits.numel() == 0:
        raise SystemExit("No samples available for calibration.")

    res = _fit_temperature(logits=logits.to(device), y=y.to(device))

    out_path = Path(args.out) if args.out else (checkpoints_dir / "temperature.json")
    if not out_path.is_absolute():
        out_path = (Path(__file__).resolve().parent.parent / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "temperature": float(res.temperature),
        "split": str(args.split),
        "n_samples": int(res.n_samples),
        "nll_before": float(res.nll_before),
        "nll_after": float(res.nll_after),
        "ece_before": float(res.ece_before),
        "ece_after": float(res.ece_after),
        "backbone": str(backbone),
        "input_size": int(input_size),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

