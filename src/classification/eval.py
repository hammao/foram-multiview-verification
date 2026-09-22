from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.classification.dataset import ViewDataset, build_label_space
from src.config.load_config import load_config
from src.data.species import load_species_list


def _ece(probs: torch.Tensor, y: torch.Tensor, *, n_bins: int = 15) -> float:
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


def _load_eval_label_names(*, checkpoints_dir: Path, species_csv: Path) -> tuple[list[str], str]:
    label_map_path = checkpoints_dir / "label_map.json"
    if label_map_path.exists():
        try:
            label_names = json.loads(label_map_path.read_text(encoding="utf-8"))
            if isinstance(label_names, list) and all(isinstance(x, str) for x in label_names):
                return list(label_names), str(label_map_path)
        except Exception:
            pass
    return load_species_list(species_csv), str(species_csv)


def evaluate(config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    species_csv = Path(cfg["paths_resolved"]["species_csv"])
    cls_cfg = cfg.get("classification", {}) if isinstance(cfg.get("classification", {}), dict) else {}

    splits_dir = Path(cfg["paths_resolved"].get("splits_dir", processed_dir / "manifest" / "splits"))
    test_csv = splits_dir / "test.csv"
    val_csv = splits_dir / "val.csv"
    train_csv = splits_dir / "train.csv"
    if not test_csv.exists():
        # Some configs intentionally create no test set (e.g., train/val only).
        test_csv = None  # type: ignore[assignment]

    split_name = "test"
    split_csv = test_csv
    if split_csv is not None:
        try:
            if split_csv.stat().st_size == 0 or pd.read_csv(split_csv).empty:
                split_csv = None
        except Exception:
            split_csv = None
    if split_csv is None:
        # Fall back to val split when test is empty/unavailable.
        split_name = "val"
        split_csv = val_csv if val_csv.exists() else None
        if split_csv is not None:
            try:
                if split_csv.stat().st_size == 0 or pd.read_csv(split_csv).empty:
                    split_csv = None
            except Exception:
                split_csv = None
    if split_csv is None:
        # Last resort: train split (avoids crashing, but not a generalization metric).
        split_name = "train"
        split_csv = train_csv if train_csv.exists() else None
        if split_csv is not None:
            try:
                if split_csv.stat().st_size == 0 or pd.read_csv(split_csv).empty:
                    split_csv = None
            except Exception:
                split_csv = None
    if split_csv is None:
        raise FileNotFoundError(f"Missing non-empty split CSV under: {splits_dir}")

    ckpt_path = checkpoints_dir / "classifier.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {ckpt_path}")

    # Prefer GPU backends when available (CUDA on NVIDIA, MPS on Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    species_list, label_source = _load_eval_label_names(checkpoints_dir=checkpoints_dir, species_csv=species_csv)
    label_space = build_label_space(species_list)
    eval_species_aliases = cls_cfg.get("eval_species_aliases", {})
    if not isinstance(eval_species_aliases, dict):
        eval_species_aliases = {}
    eval_species_aliases = {str(k): str(v) for k, v in eval_species_aliases.items()}
    reverse_aliases: dict[str, str] = {}
    for current_name, checkpoint_name in eval_species_aliases.items():
        reverse_aliases.setdefault(checkpoint_name, current_name)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    backbone = ckpt["backbone"]
    input_size = int(ckpt.get("input_size", cfg.get("resize", {}).get("input_size", 384)))
    temp_path = checkpoints_dir / "temperature.json"
    temperature = 1.0
    if temp_path.exists():
        try:
            temperature = float(json.loads(temp_path.read_text(encoding="utf-8")).get("temperature", 1.0))
        except Exception:
            temperature = 1.0

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

    split_csv_for_eval = split_csv
    temp_eval_csv: Path | None = None
    if eval_species_aliases:
        df_eval = pd.read_csv(split_csv)
        if "species" in df_eval.columns:
            df_eval["species"] = df_eval["species"].astype(str).map(lambda s: eval_species_aliases.get(s, s))
        missing = sorted(set(df_eval["species"].astype(str)) - set(species_list))
        if missing:
            raise ValueError(
                "Evaluation labels are missing from checkpoint label space after alias mapping: "
                + ", ".join(missing)
            )
        manifest_dir = processed_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            prefix="_eval_alias_",
            dir=manifest_dir,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            df_eval.to_csv(tmp.name, index=False)
            temp_eval_csv = Path(tmp.name)
        split_csv_for_eval = temp_eval_csv

    ds = ViewDataset(
        split_csv_for_eval,
        label_space=label_space,
        transform=tf,
        root=cfg["paths_resolved"].get("base_dir"),
    )
    num_workers = 2 if device.type == "cuda" else 0
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=num_workers)

    y_true = []
    y_pred = []
    logits_all = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.numpy().tolist())
            logits_all.append(logits.detach().float().cpu())

    display_species_list = [reverse_aliases.get(name, name) for name in species_list]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(species_list))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(species_list))),
        target_names=display_species_list,
        output_dict=True,
        zero_division=0,
    )

    # Calibration diagnostics (optional; temperature scaling doesn't change argmax accuracy).
    nll = float("nan")
    ece = float("nan")
    nll_calibrated = float("nan")
    ece_calibrated = float("nan")
    if logits_all:
        logits_cat = torch.cat(logits_all, dim=0)
        y_cat = torch.tensor(y_true, dtype=torch.long)
        ce = torch.nn.CrossEntropyLoss()
        nll = float(ce(logits_cat, y_cat).item())
        ece = _ece(torch.softmax(logits_cat, dim=1), y_cat)
        if temperature and temperature != 1.0:
            nll_calibrated = float(ce(logits_cat / float(temperature), y_cat).item())
            ece_calibrated = _ece(torch.softmax(logits_cat / float(temperature), dim=1), y_cat)

    out = {
        "backbone": backbone,
        "input_size": input_size,
        "split_used": str(split_name),
        "split_csv": str(split_csv),
        "label_space_source": str(label_source),
        "species_aliases_used": eval_species_aliases,
        "temperature": float(temperature),
        # Use JSON-safe values (null instead of NaN/Inf).
        "nll": float(nll) if np.isfinite(nll) else None,
        "ece": float(ece) if np.isfinite(ece) else None,
        "nll_calibrated": float(nll_calibrated) if np.isfinite(nll_calibrated) else None,
        "ece_calibrated": float(ece_calibrated) if np.isfinite(ece_calibrated) else None,
        "confusion_matrix": cm.tolist(),
        "report": report,
    }

    eval_report_name = str(cls_cfg.get("eval_report_filename", "eval_report.json")).strip() or "eval_report.json"
    out_path = processed_dir / "manifest" / eval_report_name
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pred_rows = []
    if logits_all:
        confs = torch.softmax(logits_cat, dim=1)
        top_conf = confs.gather(1, torch.tensor(y_pred).view(-1, 1)).squeeze(1).tolist()
    else:
        top_conf = [None] * len(y_true)
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        row = ds.df.iloc[i]
        pred_rows.append(
            {
                "path_resized": row.get(ds.image_col),
                "species": display_species_list[int(yt)],
                "predicted": display_species_list[int(yp)],
                "correct": int(yt == yp),
                "confidence": top_conf[i],
                "specimen_id": row.get("specimen_id"),
                "view_id": row.get("view_id"),
            }
        )
    pred_path = processed_dir / "manifest" / eval_report_name.replace(".json", "_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    out["predictions_csv"] = str(pred_path)
    if temp_eval_csv is not None:
        try:
            temp_eval_csv.unlink()
        except Exception:
            pass
    return out


if __name__ == "__main__":
    out = evaluate()
    print(json.dumps({"backbone": out["backbone"], "input_size": out["input_size"]}, indent=2))

