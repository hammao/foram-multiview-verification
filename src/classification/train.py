from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.classification.dataset import ViewDataset, build_label_space
from src.config.load_config import load_config
from src.data.species import load_species_list


@dataclass(frozen=True)
class TrainOutput:
    checkpoint_path: Path
    label_map_path: Path


def resolve_seed(cfg: dict) -> int:
    """
    Training seed, read from `classification.seed` and falling back to `split.seed`.

    Kept separate from the split seed's *use* so that changing the training seed never
    repartitions the data, but defaulting to it means existing configs stay reproducible.
    """
    cls_cfg = cfg.get("classification", {}) or {}
    if cls_cfg.get("seed", None) is not None:
        return int(cls_cfg["seed"])
    split_cfg = cfg.get("split", {}) or {}
    return int(split_cfg.get("seed", 1337))


def set_global_seed(seed: int) -> None:
    """
    Seed every generator the training loop draws from.

    Without this, model head init, RandAugment, Random Erasing and the
    WeightedRandomSampler all draw from unseeded global RNGs, so two runs of the same
    config are not comparable and `seed` in the config only ever affected the split.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        try:
            mps.manual_seed(seed)
        except Exception:
            pass


def seeded_generator(seed: int) -> torch.Generator:
    """CPU generator for DataLoader shuffling and the WeightedRandomSampler draw."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_worker(worker_id: int) -> None:
    """Re-seed numpy/random inside DataLoader workers, which fork with a fresh state."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _species_to_genus(species: str) -> str:
    s = (species or "").strip()
    if not s:
        return ""
    return s.split()[0]


def _make_genus_soft_target_matrix(idx_to_species: list[str], *, same_genus_mass: float) -> torch.Tensor:
    """
    Build a [C,C] matrix of soft targets where each true class y maps to a distribution:
    - (1 - same_genus_mass) on y
    - same_genus_mass distributed across other classes that share the same genus
      (fallback: distributed uniformly across all other classes if there are no same-genus peers)

    This encodes the idea that confusing species within the same genus is "less wrong" than
    confusing across genera, while still training a single species-level classifier.
    """
    same_genus_mass = float(same_genus_mass)
    if not (0.0 <= same_genus_mass < 1.0):
        raise ValueError("same_genus_mass must satisfy 0 <= mass < 1")

    C = len(idx_to_species)
    if C == 0:
        return torch.empty((0, 0), dtype=torch.float32)

    genera = [_species_to_genus(s) for s in idx_to_species]
    M = torch.zeros((C, C), dtype=torch.float32)
    for y in range(C):
        M[y, y] = 1.0 - same_genus_mass
        same = [i for i in range(C) if i != y and genera[i] == genera[y] and genera[y] != ""]
        if same:
            w = same_genus_mass / float(len(same))
            for i in same:
                M[y, i] = w
        else:
            # no same-genus peers (or genus missing) -> fallback uniform over other classes
            if C > 1:
                w = same_genus_mass / float(C - 1)
                for i in range(C):
                    if i != y:
                        M[y, i] = w
    return M


def resolve_amp(use_amp: bool, device: torch.device) -> tuple[torch.amp.GradScaler, str, bool]:
    """
    Return (scaler, autocast_device_type, amp_active) and say out loud when AMP is dropped.

    Mixed precision is only wired for CUDA. Previously the scaler was constructed with
    `enabled=use_amp and device.type == "cuda"`, so on MPS a config asking for AMP got
    float32 with no indication anywhere in the logs or the saved artifacts. Runs were
    reported as mixed precision when they were not.
    """
    amp_active = bool(use_amp) and device.type == "cuda"
    if use_amp and not amp_active:
        print(
            f"[amp] use_amp=true requested but mixed precision is only implemented for CUDA; "
            f"device is {device.type}. Training in float32."
        )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_active)
    return scaler, "cuda", amp_active


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    # Assume 1 param group (true here); fall back safely otherwise.
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _make_transforms(input_size: int, is_train: bool, augmentation: str):
    import timm

    # Let timm pick good defaults for the given backbone family.
    # We'll override only the input size and basic augmentation policy.
    if is_train:
        if augmentation == "rand_augment":
            aa = "rand-m9-mstd0.5-inc1"
        else:
            # Keep this conservative; timm's supported AA strings vary by version.
            # We still get random resize/crop and random erasing from timm defaults.
            aa = None
    else:
        aa = None

    return timm.data.create_transform(
        input_size=(3, input_size, input_size),
        is_training=is_train,
        auto_augment=aa,
        re_prob=0.25 if is_train else 0.0,
        re_mode="pixel",
    )


def train_from_config(config_path: str = "configs/default.yaml") -> TrainOutput:
    cfg = load_config(config_path)
    seed = resolve_seed(cfg)
    set_global_seed(seed)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    checkpoints_dir = Path(cfg["paths_resolved"]["checkpoints_dir"])
    species_csv = Path(cfg["paths_resolved"]["species_csv"])

    splits_dir = Path(cfg["paths_resolved"]["splits_dir"])
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")

    # Prefer GPU backends when available (CUDA on NVIDIA, MPS on Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    species_list = load_species_list(species_csv)
    label_space = build_label_space(species_list)

    input_size = int(cfg.get("resize", {}).get("input_size", cfg.get("classification", {}).get("input_size", 384)))
    cls_cfg = cfg.get("classification", {})
    backbone = str(cls_cfg.get("backbone", "convnext_base.fb_in22k_ft_in1k"))
    batch_size = int(cls_cfg.get("batch_size", 24))
    lr = float(cls_cfg.get("lr", 3e-4))
    wd = float(cls_cfg.get("weight_decay", 0.05))
    num_epochs = int(cls_cfg.get("num_epochs", 30))
    use_amp = bool(cls_cfg.get("use_amp", True))
    augmentation = str(cls_cfg.get("augmentation", "rand_augment"))
    grad_clip_norm = float(cls_cfg.get("grad_clip_norm", 0.0)) if cls_cfg.get("grad_clip_norm") is not None else 0.0
    label_smoothing = float(cls_cfg.get("label_smoothing", 0.0)) if cls_cfg.get("label_smoothing") is not None else 0.0
    drop_path_rate = float(cls_cfg.get("drop_path_rate", 0.0)) if cls_cfg.get("drop_path_rate") is not None else 0.0
    early_stopping_patience = int(cls_cfg.get("early_stopping_patience", 0)) if cls_cfg.get("early_stopping_patience") is not None else 0
    save_each_epoch = bool(cls_cfg.get("save_each_epoch", False))
    history_dir_name = str(cls_cfg.get("save_history_dir", "history")).strip() or "history"
    history_dir = checkpoints_dir / history_dir_name
    training_metrics_name = str(cls_cfg.get("training_metrics_filename", "training_metrics.csv")).strip() or "training_metrics.csv"
    genus_cfg = cls_cfg.get("genus_aware", {}) if isinstance(cls_cfg.get("genus_aware", {}), dict) else {}
    genus_enabled = bool(genus_cfg.get("enabled", False))
    same_genus_mass = float(genus_cfg.get("same_genus_mass", 0.15))
    # Optional modality filtering for ablations (no need to rebuild the manifest).
    # Example: ["stacked_tiff", "gif_original"] to exclude zstack slices.
    sources_include = cls_cfg.get("data_sources_include", None)
    sources_exclude = cls_cfg.get("data_sources_exclude", None)
    if isinstance(sources_include, str):
        sources_include = [sources_include]
    if isinstance(sources_exclude, str):
        sources_exclude = [sources_exclude]
    sources_include = [str(s) for s in sources_include] if isinstance(sources_include, list) else None
    sources_exclude = [str(s) for s in sources_exclude] if isinstance(sources_exclude, list) else None

    import timm

    kwargs = {"pretrained": True, "num_classes": len(species_list)}
    if drop_path_rate > 0:
        try:
            kwargs["drop_path_rate"] = drop_path_rate
        except Exception:
            pass
    model = timm.create_model(backbone, **kwargs)
    model.to(device)

    train_tf = _make_transforms(input_size=input_size, is_train=True, augmentation=augmentation)
    eval_tf = _make_transforms(input_size=input_size, is_train=False, augmentation=augmentation)

    train_ds = ViewDataset(train_csv, label_space=label_space, transform=train_tf)
    if ("source" in train_ds.df.columns) and (sources_include is not None or sources_exclude is not None):
        m = pd.Series([True] * len(train_ds.df))
        if sources_include is not None:
            m = m & train_ds.df["source"].astype(str).isin(set(sources_include))
        if sources_exclude is not None:
            m = m & (~train_ds.df["source"].astype(str).isin(set(sources_exclude)))
        train_ds.df = train_ds.df.loc[m].reset_index(drop=True)
        if train_ds.df.empty:
            raise ValueError(f"Training split became empty after source filtering include={sources_include} exclude={sources_exclude}")
    # DataLoader tuning:
    # - On CUDA: workers + pin_memory help
    # - On MPS/CPU (macOS): keep it conservative to avoid overhead/stalls
    if device.type == "cuda":
        num_workers = 2
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # Sampling strategy: when we ingest rich sources (especially Z-stack slices),
    # naive shuffling can over-represent sources with many near-duplicate frames.
    sampling_cfg = cls_cfg.get("sampling", {}) if isinstance(cls_cfg.get("sampling", {}), dict) else {}
    sampling_strategy = str(sampling_cfg.get("strategy", "none")).lower().strip()
    sampler = None
    if sampling_strategy in {"balanced_specimen_view", "balanced"}:
        # Balance per (species, specimen_id, view_id) so each physical angle contributes similarly,
        # regardless of whether it has 1 stacked image or 40 Z-slices.
        group_cols = sampling_cfg.get("group_cols", ["species", "specimen_id", "view_id"])
        if not isinstance(group_cols, list):
            group_cols = ["species", "specimen_id", "view_id"]
        group_cols = [str(c) for c in group_cols]
        if all(c in train_ds.df.columns for c in group_cols):
            g = train_ds.df.groupby(group_cols, sort=False).size().rename("group_n").reset_index()
            df_w = train_ds.df.reset_index(drop=True).merge(g, on=group_cols, how="left")
            # Weight each row as 1 / group_size => sum weights per group ~= 1
            w = (1.0 / df_w["group_n"].astype(float)).astype(float).to_numpy()
            num_samples = sampling_cfg.get("num_samples_per_epoch", None)
            num_samples = int(num_samples) if num_samples is not None else int(len(train_ds))
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(w, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=seeded_generator(seed),
            )
        else:
            sampler = None
    # If sampler is set, shuffle must be False.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=seeded_generator(seed),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    val_loader = None
    if val_csv.exists() and val_csv.stat().st_size > 0:
        try:
            val_ds = ViewDataset(val_csv, label_space=label_space, transform=eval_tf)
            if ("source" in val_ds.df.columns) and (sources_include is not None or sources_exclude is not None):
                m = pd.Series([True] * len(val_ds.df))
                if sources_include is not None:
                    m = m & val_ds.df["source"].astype(str).isin(set(sources_include))
                if sources_exclude is not None:
                    m = m & (~val_ds.df["source"].astype(str).isin(set(sources_exclude)))
                val_ds.df = val_ds.df.loc[m].reset_index(drop=True)
                if val_ds.df.empty:
                    # Keep val_loader=None rather than crash; training will fall back to train-only checkpointing.
                    raise ValueError("Validation split became empty after source filtering.")
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        except Exception:
            val_loader = None

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    genus_target_matrix = None
    genus_ids = None
    if genus_enabled:
        genus_target_matrix = _make_genus_soft_target_matrix(label_space.idx_to_species, same_genus_mass=same_genus_mass).to(device)
        genera = [_species_to_genus(s) for s in label_space.idx_to_species]
        genus_to_id: dict[str, int] = {}
        genus_ids = torch.tensor([genus_to_id.setdefault(g, len(genus_to_id)) for g in genera], dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched_cfg = cls_cfg.get("lr_schedule", cls_cfg.get("scheduler", "none"))
    lr_schedule = str(sched_cfg).lower().strip() if sched_cfg is not None else "none"
    lr_min = float(cls_cfg.get("lr_min", cls_cfg.get("min_lr", 1e-6)))
    scheduler = None
    if lr_schedule in {"cosine", "cosineannealing", "cosine_annealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    scaler, amp_device_type, amp_active = resolve_amp(use_amp, device)

    # Training metrics log (publication-friendly).
    # Written under processed/manifest/ so report packs can include it.
    metrics_path = processed_dir / "manifest" / training_metrics_name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, float | int | str]] = []

    def eval_acc() -> float:
        if val_loader is None:
            return float("nan")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        model.train()
        return correct / max(1, total)

    def eval_genus_acc() -> float:
        if val_loader is None or genus_ids is None:
            return float("nan")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += int((genus_ids[pred] == genus_ids[yb]).sum().item())
                total += int(yb.numel())
        model.train()
        return correct / max(1, total)

    best_acc = -1.0
    epochs_without_improvement = 0
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / "classifier.pt"
    if save_each_epoch:
        history_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, num_epochs + 1):
        t0 = perf_counter()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}", leave=False)
        running = 0.0
        correct = 0
        correct_genus = 0
        total = 0
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_active):
                logits = model(xb)
                if genus_target_matrix is None:
                    loss = criterion(logits, yb)
                else:
                    # genus-aware soft targets: loss = -sum(target * log_softmax(logits))
                    logp = torch.log_softmax(logits, dim=1)
                    tgt = genus_target_matrix[yb]  # [B,C]
                    loss = -(tgt * logp).sum(dim=1).mean()
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            if genus_ids is not None:
                correct_genus += int((genus_ids[pred] == genus_ids[yb]).sum().item())
            total += int(yb.numel())
            running += float(loss.item())
            pbar.set_postfix(loss=running / max(1, pbar.n + 1))

        acc = eval_acc()
        genus_acc = eval_genus_acc()
        train_loss = running / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_genus_acc = (correct_genus / max(1, total)) if genus_ids is not None else float("nan")
        lr_now = _optimizer_lr(optimizer)
        epoch_s = perf_counter() - t0

        # Step LR scheduler once per epoch (after optimizer steps).
        if scheduler is not None:
            scheduler.step()

        metrics_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_genus_acc": float(train_genus_acc),
                "val_acc": float(acc),
                "val_genus_acc": float(genus_acc),
                "lr": float(lr_now),
                "epoch_seconds": float(epoch_s),
            }
        )
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

        if genus_ids is None:
            print(f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={acc:.4f} lr={lr_now:.6g}")
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_genus_acc={train_genus_acc:.4f} "
                f"val_acc={acc:.4f} val_genus_acc={genus_acc:.4f} lr={lr_now:.6g}"
            )
        if save_each_epoch:
            # Keep an epoch snapshot so validation-only selection can rank a complete history.
            # Written before the early-stopping break, or the stopping epoch is never saved
            # and the history is one checkpoint short of the metrics log.
            torch.save(
                {"model": model.state_dict(), "backbone": backbone, "input_size": input_size, "epoch": int(epoch)},
                history_dir / f"classifier_epoch{epoch:03d}.pt",
            )

        if val_loader is not None and acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0
            torch.save({"model": model.state_dict(), "backbone": backbone, "input_size": input_size}, ckpt_path)
        elif val_loader is not None and early_stopping_patience > 0:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {early_stopping_patience} epochs).")
                break

    # If there's no validation split (train-all mode), ALWAYS write a final checkpoint.
    # Otherwise, keep the best-by-val checkpoint written above, and only fall back to final
    # if none was written (e.g., empty/invalid val split).
    if val_loader is None:
        torch.save({"model": model.state_dict(), "backbone": backbone, "input_size": input_size}, ckpt_path)
    elif not ckpt_path.exists():
        torch.save({"model": model.state_dict(), "backbone": backbone, "input_size": input_size}, ckpt_path)

    label_map_path = checkpoints_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_space.idx_to_species, indent=2), encoding="utf-8")

    return TrainOutput(checkpoint_path=ckpt_path, label_map_path=label_map_path)


if __name__ == "__main__":
    out = train_from_config()
    print(f"checkpoint: {out.checkpoint_path}")
    print(f"labels: {out.label_map_path}")

