"""Penultimate-layer embeddings of the deployed single-stream classifier for the frozen test tier.

Writes reports/experiments/2026-10_drive140_paper/<arm>/embeddings_test.npz (1024-d features)
and embedding_pca_test.csv (PC1, PC2, species, specimen_id, source, correct). PCA is fitted on
the test embeddings themselves; the figure is descriptive and takes no part in any decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from paper_score_and_summarize import TABLE1, device  # noqa: E402
from src.classification.transforms import build_eval_transform  # noqa: E402
from src.config.load_config import load_config  # noqa: E402

REP = ROOT / "reports" / "experiments" / "2026-10_drive140_paper"
SPLIT = ROOT / "processed" / "drive140" / "manifest" / "splits_specimen_tvt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/train_single_stream_drive140_paper.yaml")
    ap.add_argument("--arm", default="single_stream")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    ckpt = torch.load(ckpt_dir / "classifier.pt", map_location="cpu")
    labels = json.loads((ckpt_dir / "label_map.json").read_text(encoding="utf-8"))
    import timm

    model = timm.create_model(ckpt["backbone"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    dev = device()
    model.to(dev).eval()
    tf = build_eval_transform(int(ckpt.get("input_size", 384)))
    from PIL import Image

    df = pd.read_csv(SPLIT / f"{args.split}.csv")
    feats, preds = [], []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.path_resized.iloc[i : i + 32].astype(str)]).to(dev)
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            feats.append(f.float().cpu())
            preds.append(model.forward_head(model.forward_features(x)).argmax(1).cpu())
    X = torch.cat(feats).numpy()
    pred = [labels[i] for i in torch.cat(preds).numpy()]

    Xc = X - X.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Xc @ vt[:2].T
    var = (s**2) / (s**2).sum()

    tax = pd.read_csv(TABLE1).set_index("code_label")
    out = pd.DataFrame({
        "species": df.species, "specimen_id": df.specimen_id, "view_id": df.view_id, "source": df.source,
        "display_name": df.species.map(tax.display_name), "wall_type": df.species.map(tax.wall_type), "family": df.species.map(tax.family),
        "pred_1": pred, "correct": (df.species.values == np.array(pred)).astype(int), "PC1": pcs[:, 0], "PC2": pcs[:, 1],
    })
    out_dir = REP / args.arm
    np.savez_compressed(out_dir / f"embeddings_{args.split}.npz", X=X, species=df.species.to_numpy(), specimen_id=df.specimen_id.to_numpy())
    out.to_csv(out_dir / f"embedding_pca_{args.split}.csv", index=False)
    (out_dir / f"embedding_pca_{args.split}_meta.json").write_text(json.dumps({"n": int(len(df)), "dim": int(X.shape[1]), "explained_variance_pc1": float(var[0]), "explained_variance_pc2": float(var[1])}, indent=2), encoding="utf-8")
    print(f"{len(df)} embeddings of dim {X.shape[1]}; PC1 {var[0]:.3f}, PC2 {var[1]:.3f} of variance -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
