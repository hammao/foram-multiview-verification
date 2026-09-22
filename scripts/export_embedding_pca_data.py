#!/usr/bin/env python3
"""
Export 2D PCA of classifier penultimate features to CSV for reproducibility and R plotting.

Reads the resized manifest, loads the deployed classifier, extracts features for a sampled
subset of images, fits PCA(n_components=2), and writes:
  - embedding_pca_data.csv: path_resized, species, genus, specimen_id, view_id, source, PC1, PC2
  - embedding_pca_metadata.json: n_components, random_state, n_points, backbone, input_size, etc.

Usage:
  python scripts/export_embedding_pca_data.py --config configs/train_genus_autocrop_with_val.yaml
  python scripts/export_embedding_pca_data.py --config configs/train_genus_autocrop_with_val.yaml --max-points 200 --out-dir processed/embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.load_config import load_config


def _species_to_genus(species: str) -> str:
    s = (species or "").strip()
    return s.split()[0] if s else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Export PCA embedding data for reproducibility.")
    ap.add_argument("--config", type=str, default="configs/train_genus_autocrop_with_val.yaml")
    ap.add_argument("--max-points", type=int, default=120)
    ap.add_argument("--out-dir", type=str, default="processed/embeddings")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_dir = Path(cfg["paths_resolved"]["checkpoints_dir"]).resolve().parent
    manifest_csv = base_dir / "processed" / "manifest" / "views_resized.csv"
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"]).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        print(f"Manifest not found: {manifest_csv}", file=sys.stderr)
        sys.exit(1)

    df_views = pd.read_csv(manifest_csv)
    df_s = (
        df_views.dropna(subset=["path_resized", "species"])
        .sample(n=min(args.max_points, len(df_views)), random_state=args.seed)
    )

    ckpt_path = checkpoints_dir / "classifier.pt"
    labels_path = checkpoints_dir / "label_map.json"
    if not ckpt_path.exists() or not labels_path.exists():
        print(f"Checkpoint or label_map not found in {checkpoints_dir}", file=sys.stderr)
        sys.exit(1)

    import torch
    import torchvision.transforms as T
    from sklearn.decomposition import PCA

    try:
        torch.set_num_threads(min(4, __import__("os").cpu_count() or 4))
    except Exception:
        pass

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = str(ckpt.get("backbone", ""))
    input_size = int(ckpt.get("input_size", 384))
    n_classes = len(json.loads(labels_path.read_text()))

    import timm
    model = timm.create_model(backbone, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model.to(device)

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    feats = []
    rows = []
    for _, row in df_s.iterrows():
        p = Path(row["path_resized"])
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im = ImageOps.pad(im, (input_size, input_size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))
        x = tf(im).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                f = model.forward_features(x)
                if isinstance(f, (tuple, list)):
                    f = f[0]
                if f.ndim == 4:
                    f = f.mean(dim=(2, 3))
            else:
                f = model(x)
            f = f.detach().float().cpu().numpy().reshape(-1)
        feats.append(f)
        sp = str(row["species"])
        rows.append({
            "path_resized": str(p),
            "species": sp,
            "genus": _species_to_genus(sp),
            "specimen_id": row.get("specimen_id", ""),
            "view_id": row.get("view_id", ""),
            "source": row.get("source", ""),
        })

    if len(feats) < 3:
        print("Too few valid images for PCA.", file=sys.stderr)
        sys.exit(1)

    X = np.stack(feats, axis=0)
    pca = PCA(n_components=2, random_state=args.seed)
    X2 = pca.fit_transform(X)

    out_df = pd.DataFrame(rows)
    out_df["PC1"] = X2[:, 0]
    out_df["PC2"] = X2[:, 1]

    csv_path = out_dir / "embedding_pca_data.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(out_df)} points)")

    meta = {
        "n_components": 2,
        "random_state": args.seed,
        "n_points": len(out_df),
        "backbone": backbone,
        "input_size": input_size,
        "n_species": int(out_df["species"].nunique()),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    meta_path = out_dir / "embedding_pca_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
