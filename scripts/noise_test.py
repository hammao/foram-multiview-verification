from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config.load_config import load_config
from src.inference.pipeline import load_classifier


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _add_gaussian_noise(img: Image.Image, sigma: float, seed: int | None) -> Image.Image:
    """Add zero-mean Gaussian noise with std=sigma (in pixel space 0..255)."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _add_salt_pepper(img: Image.Image, amount: float, seed: int | None) -> Image.Image:
    """Salt & pepper noise; amount in [0..1] fraction of pixels affected."""
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    h, w, _ = arr.shape
    n = int(amount * h * w)
    if n <= 0:
        return Image.fromarray(arr, mode="RGB")
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    salt = rng.random(size=n) < 0.5
    arr[ys[salt], xs[salt], :] = 255
    arr[ys[~salt], xs[~salt], :] = 0
    return Image.fromarray(arr, mode="RGB")


def _add_stretch_distortion(img: Image.Image, factor: float, seed: int | None) -> Image.Image:
    """
    Apply anisotropic stretch/compression and resample back to original size.
    factor > 1 stretches one axis while compressing the other.
    """
    im = img.convert("RGB")
    w, h = im.size
    if w < 4 or h < 4:
        return im

    f = max(1.0, float(factor))
    rng = np.random.default_rng(seed)
    stretch_x = bool(rng.integers(0, 2))

    if stretch_x:
        w2 = max(8, int(round(w * f)))
        h2 = max(8, int(round(h / f)))
    else:
        w2 = max(8, int(round(w / f)))
        h2 = max(8, int(round(h * f)))

    warped = im.resize((w2, h2), resample=Image.Resampling.BICUBIC)
    return warped.resize((w, h), resample=Image.Resampling.BICUBIC)


@dataclass(frozen=True)
class NoiseCase:
    kind: str  # gaussian | saltpepper | stretch
    level: float


def _predict(model, labels: list[str], input_size: int, image_path: Path) -> tuple[str, float]:
    import torchvision.transforms as T
    from PIL import ImageOps
    import torch

    device = next(model.parameters()).device

    im = Image.open(image_path).convert("RGB")
    im = ImageOps.pad(im, (input_size, input_size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
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
        top_prob, top_idx = torch.max(probs, dim=0)

    return labels[int(top_idx.item())], float(top_prob.item())


def _load_samples_from_csv(csv_path: Path, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    # Prefer resized paths if present
    img_col = "path_resized" if "path_resized" in df.columns else "path"
    df = df.dropna(subset=[img_col, "species"])
    if df.empty:
        return []
    if int(n) > 0:
        df = df.sample(n=min(int(n), len(df)), random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    out = []
    for _, r in df.iterrows():
        out.append(
            {
                "species": str(r["species"]),
                "path": str(r[img_col]),
                "source": str(r.get("source", "")),
                "specimen_id": int(r.get("specimen_id", -1)),
                "view_id": int(r.get("view_id", -1)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--input", help="Optional: single image path to test.")
    ap.add_argument("--true-species", help="Required if --input is provided.")
    ap.add_argument("--from-csv", default="processed/manifest/splits/test.csv", help="CSV to sample images from.")
    ap.add_argument(
        "--n",
        type=int,
        default=0,
        help="How many samples from CSV to test. 0 (default) uses the full CSV — the 202-image test split.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gaussian-sigmas", nargs="*", type=float, default=[0, 5, 10, 25, 50])
    ap.add_argument("--saltpepper-amounts", nargs="*", type=float, default=[0.0, 0.01, 0.03, 0.07])
    ap.add_argument(
        "--stretch-factors",
        nargs="*",
        type=float,
        default=[1.0],
        help="Anisotropic stretch factors (>=1.0). Example: 1.2 1.35",
    )
    ap.add_argument("--output-dir", default="processed/noise_tests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    model, labels, input_size = load_classifier(checkpoints_dir)

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    out_dir = Path(args.output_dir).resolve()
    run_dir = out_dir / f"noise_test_{_ts()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build test cases
    cases: list[NoiseCase] = []
    for s in args.gaussian_sigmas:
        cases.append(NoiseCase(kind="gaussian", level=float(s)))
    for a in args.saltpepper_amounts:
        cases.append(NoiseCase(kind="saltpepper", level=float(a)))
    for f in args.stretch_factors:
        if float(f) >= 1.0:
            cases.append(NoiseCase(kind="stretch", level=float(f)))

    # Select samples
    samples: list[dict]
    if args.input:
        if not args.true_species:
            raise SystemExit("--true-species is required when using --input")
        samples = [
            {
                "species": args.true_species,
                "path": str(Path(args.input).resolve()),
                "source": "manual",
                "specimen_id": -1,
                "view_id": -1,
            }
        ]
    else:
        csv_path = Path(args.from_csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        samples = _load_samples_from_csv(csv_path, n=args.n, seed=args.seed)
        if not samples:
            raise SystemExit(f"No usable samples found in {csv_path}")

    # Run tests
    rows = []
    for si, s in enumerate(samples):
        true_species = s["species"]
        src_path = Path(s["path"])
        base_img = Image.open(src_path).convert("RGB")

        for ci, c in enumerate(cases):
            if c.kind == "gaussian":
                noisy = _add_gaussian_noise(base_img, sigma=c.level, seed=args.seed + si * 1000 + ci)
                tag = f"gauss_sigma{c.level:g}"
            elif c.kind == "saltpepper":
                noisy = _add_salt_pepper(base_img, amount=c.level, seed=args.seed + si * 1000 + ci)
                tag = f"sp_amt{c.level:g}"
            else:
                noisy = _add_stretch_distortion(base_img, factor=c.level, seed=args.seed + si * 1000 + ci)
                tag = f"stretch_f{c.level:g}"

            out_img = run_dir / f"s{si}_{tag}.png"
            noisy.save(out_img)

            pred, conf = _predict(model, labels, input_size, out_img)
            rows.append(
                {
                    "sample_index": si,
                    "source_path": str(src_path),
                    "true_species": true_species,
                    "noise_kind": c.kind,
                    "noise_level": c.level,
                    "predicted_species": pred,
                    "confidence": conf,
                    "correct": int(pred == true_species),
                    "out_image": str(out_img),
                }
            )

    out_csv = run_dir / "results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoints_dir": str(checkpoints_dir),
        "input_size": int(input_size),
        "num_samples": len(samples),
        "num_cases": len(cases),
        "accuracy_overall": float(np.mean([r["correct"] for r in rows])) if rows else float("nan"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

