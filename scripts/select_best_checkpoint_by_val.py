"""
Select a deployment checkpoint using the validation split only.

Replaces scripts/select_best_checkpoint_by_verified.py, which ranked epoch checkpoints by
accuracy on the human-verified audit cohort and copied the winner over the deployed
classifier.pt. That made every subsequently reported audit accuracy a selection statistic
rather than an independent estimate (reviewer comments M2 and M17).

The validation split is inside the training firewall and exists for this job. Nothing here
reads the audit cohort or the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classification.dataset import ViewDataset, build_label_space  # noqa: E402
from src.classification.train import _make_transforms  # noqa: E402
from src.config.load_config import load_config  # noqa: E402
from src.data.species import load_species_list  # noqa: E402


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, genus_of: list[int]) -> tuple[float, float]:
    model.eval()
    correct = genus_correct = total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        g = torch.as_tensor(genus_of, device=device)
        genus_correct += int((g[pred] == g[yb]).sum().item())
        total += int(yb.numel())
    if total == 0:
        return 0.0, 0.0
    return correct / total, genus_correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank epoch checkpoints on the validation split and optionally deploy the winner.")
    ap.add_argument("--config", required=True, help="Config whose paths.checkpoints_dir holds label_map.json + history/*.pt")
    ap.add_argument("--split", default="val", choices=["val"], help="Validation only. Test and audit are out of bounds by design.")
    ap.add_argument("--metric", default="val_acc", choices=["val_acc", "val_genus_acc"])
    ap.add_argument("--out-csv", default=None, help="Defaults to <checkpoints_dir>/checkpoint_selection_val.csv")
    ap.add_argument("--apply", action="store_true", help="Copy the winning checkpoint to <checkpoints_dir>/classifier.pt")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    ckpt_root = Path(cfg["paths_resolved"]["checkpoints_dir"])
    species_csv = Path(cfg["paths_resolved"]["species_csv"])

    hist_dir = ckpt_root / str((cfg.get("classification") or {}).get("save_history_dir", "history") or "history")
    if not hist_dir.exists():
        raise SystemExit(f"Missing history dir: {hist_dir}")
    ckpts = sorted(hist_dir.glob("classifier_epoch*.pt"))
    if not ckpts:
        raise SystemExit(f"No epoch checkpoints in {hist_dir}")

    val_csv = Path(cfg["paths_resolved"]["splits_dir"]) / f"{args.split}.csv"
    if not val_csv.exists():
        raise SystemExit(f"Missing split: {val_csv}")

    species_list = load_species_list(species_csv)
    label_space = build_label_space(species_list)
    cls_cfg = cfg.get("classification") or {}
    input_size = int(cfg.get("resize", {}).get("input_size", cls_cfg.get("input_size", 384)))
    eval_tf = _make_transforms(input_size=input_size, is_train=False, augmentation=str(cls_cfg.get("augmentation", "rand_augment")))
    ds = ViewDataset(val_csv, label_space=label_space, transform=eval_tf, root=cfg["paths_resolved"].get("base_dir"))
    loader = DataLoader(ds, batch_size=int(cls_cfg.get("batch_size", 24)), shuffle=False, num_workers=0)

    genus_names = [str(s).split()[0] for s in label_space.idx_to_species]
    genus_idx = {g: i for i, g in enumerate(sorted(set(genus_names)))}
    genus_of = [genus_idx[g] for g in genus_names]

    device = _device()
    import timm

    rows: list[dict] = []
    for p in ckpts:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        backbone = str(ckpt.get("backbone", "convnext_base.fb_in22k_ft_in1k"))
        model = timm.create_model(backbone, pretrained=False, num_classes=len(species_list))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        acc, genus_acc = _evaluate(model, loader, device, genus_of)
        epoch = int(ckpt.get("epoch", int(p.stem.replace("classifier_epoch", ""))))
        rows.append(
            {
                "epoch": epoch,
                "checkpoint_path": str(p),
                "backbone": backbone,
                "input_size": int(ckpt.get("input_size", input_size)),
                "training_mode": str(ckpt.get("training_mode", "single_stream")),
                "val_acc": float(acc),
                "val_genus_acc": float(genus_acc),
                "n_val": int(len(ds)),
            }
        )
        print(f"epoch={epoch:3d} val_acc={acc:.4f} val_genus_acc={genus_acc:.4f}")
        del model

    # Ties broken toward the earlier epoch: the least-trained model that achieves the
    # best validation score is the most conservative choice.
    res = pd.DataFrame(rows).sort_values([args.metric, "epoch"], ascending=[False, True]).reset_index(drop=True)
    out_csv = Path(args.out_csv) if args.out_csv else ckpt_root / "checkpoint_selection_val.csv"
    if not out_csv.is_absolute():
        out_csv = (project_root / out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    best = res.iloc[0].to_dict()
    print(json.dumps({"selected": best, "selection_metric": args.metric, "selection_split": args.split, "out_csv": str(out_csv)}, indent=2))

    if args.apply:
        src = Path(str(best["checkpoint_path"]))
        (ckpt_root / "classifier.pt").write_bytes(src.read_bytes())
        (ckpt_root / "selection_provenance.json").write_text(
            json.dumps(
                {
                    "selected_epoch": int(best["epoch"]),
                    "selected_from": str(src),
                    "selection_metric": args.metric,
                    "selection_split": args.split,
                    "selection_value": float(best[args.metric]),
                    "n_candidates": int(len(res)),
                    "audit_cohort_used": False,
                    "test_split_used": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"applied epoch {int(best['epoch'])} -> {ckpt_root / 'classifier.pt'}")


if __name__ == "__main__":
    main()
