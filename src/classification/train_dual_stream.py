from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.classification.dataset import ViewDataset, build_label_space
from src.classification.dataset_dual_stream import MultiViewDistillationDataset
from src.classification.train import (
    TrainOutput,
    _make_genus_soft_target_matrix,
    _make_transforms,
    _optimizer_lr,
    _species_to_genus,
    resolve_amp,
    resolve_seed,
    seed_worker,
    seeded_generator,
    set_global_seed,
)
from src.config.load_config import load_config
from src.data.species import load_species_list


#: Recorded in every checkpoint and experiment_meta.json this trainer writes.
TRAINING_MODE = "dual_stream"

#: Value written before the 2026-09 rename. Checkpoints archived under
#: 02_V1_PROVENANCE/ and 03_ML_EXPERIMENTS/ still carry it, so readers must accept both.
LEGACY_TRAINING_MODES = ("multiview_distillation",)


def is_dual_stream_checkpoint(ckpt: dict) -> bool:
    mode = str(ckpt.get("training_mode", ""))
    return mode == TRAINING_MODE or mode in LEGACY_TRAINING_MODES


def _extract_prelogits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    feats = model.forward_features(x)
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]

    if hasattr(model, "forward_head"):
        try:
            pre_logits = model.forward_head(feats, pre_logits=True)
            if torch.is_tensor(pre_logits):
                return pre_logits
        except TypeError:
            pass
        except Exception:
            pass

    if not torch.is_tensor(feats):
        raise TypeError("Model forward_features did not return a tensor.")
    if feats.ndim == 4:
        return feats.mean(dim=(-2, -1))
    if feats.ndim == 3:
        return feats.mean(dim=1)
    return feats


def _classification_loss(
    logits: torch.Tensor,
    yb: torch.Tensor,
    criterion: nn.Module,
    *,
    genus_target_matrix: torch.Tensor | None,
) -> torch.Tensor:
    if genus_target_matrix is None:
        return criterion(logits, yb)
    logp = torch.log_softmax(logits, dim=1)
    tgt = genus_target_matrix[yb]
    return -(tgt * logp).sum(dim=1).mean()


def _apply_view_dropout(mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return mask
    keep = torch.rand(mask.shape, device=mask.device) >= float(drop_prob)
    out = mask & keep
    empty_rows = out.sum(dim=1) == 0
    if torch.any(empty_rows):
        fallback_idx = mask[empty_rows].float().argmax(dim=1)
        out_empty = out[empty_rows]
        out_empty[torch.arange(out_empty.shape[0], device=mask.device), fallback_idx] = True
        out[empty_rows] = out_empty
    return out


class MultiViewDistillationModel(nn.Module):
    def __init__(
        self,
        *,
        student_model: nn.Module,
        num_classes: int,
        num_view_embeddings: int,
        teacher_num_heads: int,
        teacher_num_layers: int,
        teacher_ff_mult: float,
        teacher_dropout: float,
        teacher_token_chunk_size: int,
    ) -> None:
        super().__init__()
        self.student_model = student_model
        self.feature_dim = int(getattr(student_model, "num_features", 0) or 1024)
        ff_dim = max(self.feature_dim, int(round(self.feature_dim * float(teacher_ff_mult))))
        self.view_embed = nn.Embedding(int(num_view_embeddings), self.feature_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=int(teacher_num_heads),
            dim_feedforward=ff_dim,
            dropout=float(teacher_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.teacher_fusion = nn.TransformerEncoder(enc_layer, num_layers=int(teacher_num_layers))
        self.teacher_norm = nn.LayerNorm(self.feature_dim)
        self.teacher_head = nn.Linear(self.feature_dim, int(num_classes))
        self.teacher_token_chunk_size = max(1, int(teacher_token_chunk_size))

    def forward_student(self, xb: torch.Tensor) -> torch.Tensor:
        return self.student_model(xb)

    def forward_teacher(
        self,
        teacher_imgs: torch.Tensor,
        teacher_mask: torch.Tensor,
        teacher_view_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, V, C, H, W = teacher_imgs.shape
        flat_imgs = teacher_imgs.view(B * V, C, H, W)
        flat_mask = teacher_mask.view(B * V)
        feats = teacher_imgs.new_zeros((B * V, self.feature_dim))

        if torch.any(flat_mask):
            valid_imgs = flat_imgs[flat_mask]
            chunks: list[torch.Tensor] = []
            for start in range(0, valid_imgs.shape[0], self.teacher_token_chunk_size):
                chunk = valid_imgs[start : start + self.teacher_token_chunk_size]
                # Recompute backbone activations in backward so peak memory is one
                # chunk, not all teacher views at once. Required on MPS.
                if chunk.requires_grad is False:
                    chunk = chunk.detach().requires_grad_(True)
                chunks.append(
                    checkpoint(_extract_prelogits, self.student_model, chunk, use_reentrant=False)
                )
            feats_valid = torch.cat(chunks, dim=0)
            feats[flat_mask] = feats_valid.to(dtype=feats.dtype)

        feats = feats.view(B, V, self.feature_dim)
        view_tokens = feats + self.view_embed(teacher_view_ids.clamp_min(0))
        fused = self.teacher_fusion(view_tokens, src_key_padding_mask=(~teacher_mask))
        mask_f = teacher_mask.unsqueeze(-1).to(dtype=fused.dtype)
        pooled = (fused * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        pooled = self.teacher_norm(pooled)
        return self.teacher_head(pooled)


def train_from_config(config_path: str = "configs/train_dual_stream.yaml") -> TrainOutput:
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cls_cfg = cfg.get("classification", {}) if isinstance(cfg.get("classification", {}), dict) else {}
    # Teacher hyperparameters. `multiview_distillation` is the pre-rename key name and is
    # still accepted so archived configs keep working.
    mvd_cfg = cls_cfg.get("dual_stream", cls_cfg.get("multiview_distillation", {}))
    if not isinstance(mvd_cfg, dict):
        mvd_cfg = {}

    species_list = load_species_list(species_csv)
    label_space = build_label_space(species_list)

    input_size = int(cfg.get("resize", {}).get("input_size", cls_cfg.get("input_size", 384)))
    backbone = str(cls_cfg.get("backbone", "convnext_base.fb_in22k_ft_in1k"))
    batch_size = int(cls_cfg.get("batch_size", 8))
    accum_steps = max(1, int(cls_cfg.get("grad_accum_steps", 1)))
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
    training_metrics_name = str(cls_cfg.get("training_metrics_filename", "training_metrics_dual_stream.csv")).strip()
    training_metrics_name = training_metrics_name or "training_metrics_dual_stream.csv"

    genus_cfg = cls_cfg.get("genus_aware", {}) if isinstance(cls_cfg.get("genus_aware", {}), dict) else {}
    genus_enabled = bool(genus_cfg.get("enabled", False))
    same_genus_mass = float(genus_cfg.get("same_genus_mass", 0.15))

    source_priority = mvd_cfg.get("source_priority", ["stacked_tiff", "gif_original", "zstack_slice"])
    if not isinstance(source_priority, list):
        source_priority = ["stacked_tiff", "gif_original", "zstack_slice"]
    teacher_view_ids = mvd_cfg.get("teacher_view_ids", None)
    if isinstance(teacher_view_ids, list):
        teacher_view_ids = [int(v) for v in teacher_view_ids]
    else:
        teacher_view_ids = None
    zstack_selection = str(mvd_cfg.get("zstack_selection", "median"))
    teacher_num_heads = int(mvd_cfg.get("teacher_num_heads", 4))
    teacher_num_layers = int(mvd_cfg.get("teacher_num_layers", 1))
    teacher_ff_mult = float(mvd_cfg.get("teacher_ff_mult", 2.0))
    teacher_dropout = float(mvd_cfg.get("teacher_dropout", 0.1))
    teacher_loss_weight = float(mvd_cfg.get("teacher_loss_weight", 1.0))
    distill_loss_weight = float(mvd_cfg.get("distill_loss_weight", 0.5))
    distill_temperature = float(mvd_cfg.get("distill_temperature", 2.0))
    teacher_view_dropout_prob = float(mvd_cfg.get("teacher_view_dropout_prob", 0.0))
    teacher_token_chunk_size = int(mvd_cfg.get("teacher_token_chunk_size", 64))

    import timm

    kwargs = {"pretrained": True, "num_classes": len(species_list)}
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    student_model = timm.create_model(backbone, **kwargs)

    train_tf = _make_transforms(input_size=input_size, is_train=True, augmentation=augmentation)
    eval_tf = _make_transforms(input_size=input_size, is_train=False, augmentation=augmentation)

    train_ds = MultiViewDistillationDataset(
        train_csv,
        label_space=label_space,
        student_transform=train_tf,
        teacher_transform=eval_tf,
        teacher_view_ids=teacher_view_ids,
        source_priority=[str(s) for s in source_priority],
        zstack_selection=zstack_selection,
    )

    if device.type == "cuda":
        num_workers = 2
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    sampling_cfg = cls_cfg.get("sampling", {}) if isinstance(cls_cfg.get("sampling", {}), dict) else {}
    sampling_strategy = str(sampling_cfg.get("strategy", "none")).lower().strip()
    sampler = None
    if sampling_strategy in {"balanced_specimen_view", "balanced"}:
        group_cols = sampling_cfg.get("group_cols", ["species", "specimen_id", "view_id"])
        if not isinstance(group_cols, list):
            group_cols = ["species", "specimen_id", "view_id"]
        group_cols = [str(c) for c in group_cols]
        if all(c in train_ds.df.columns for c in group_cols):
            g = train_ds.df.groupby(group_cols, sort=False).size().rename("group_n").reset_index()
            df_w = train_ds.df.reset_index(drop=True).merge(g, on=group_cols, how="left")
            weights = (1.0 / df_w["group_n"].astype(float)).astype(float).to_numpy()
            num_samples = sampling_cfg.get("num_samples_per_epoch", None)
            num_samples = int(num_samples) if num_samples is not None else int(len(train_ds))
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=num_samples,
                replacement=True,
                generator=seeded_generator(seed),
            )

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
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        except Exception:
            val_loader = None

    num_view_embeddings = max(train_ds.teacher_view_ids) + 1
    model = MultiViewDistillationModel(
        student_model=student_model,
        num_classes=len(species_list),
        num_view_embeddings=num_view_embeddings,
        teacher_num_heads=teacher_num_heads,
        teacher_num_layers=teacher_num_layers,
        teacher_ff_mult=teacher_ff_mult,
        teacher_dropout=teacher_dropout,
        teacher_token_chunk_size=teacher_token_chunk_size,
    ).to(device)

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
                logits = model.forward_student(xb)
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
                logits = model.forward_student(xb)
                pred = logits.argmax(dim=1)
                correct += int((genus_ids[pred] == genus_ids[yb]).sum().item())
                total += int(yb.numel())
        model.train()
        return correct / max(1, total)

    best_acc = -1.0
    epochs_without_improvement = 0
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoints_dir / "classifier.pt"
    aux_path = checkpoints_dir / "teacher_aux.pt"
    if save_each_epoch:
        history_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, num_epochs + 1):
        t0 = perf_counter()
        pbar = tqdm(train_loader, desc=f"mvd epoch {epoch}/{num_epochs}", leave=False)
        running_total = 0.0
        running_student = 0.0
        running_teacher = 0.0
        running_distill = 0.0
        correct = 0
        correct_genus = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        for step_i, (xb, yb, teacher_imgs, teacher_mask, teacher_view_ids_tensor) in enumerate(pbar, start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            teacher_imgs = teacher_imgs.to(device)
            teacher_mask = teacher_mask.to(device)
            teacher_view_ids_tensor = teacher_view_ids_tensor.to(device)

            if teacher_view_dropout_prob > 0:
                teacher_mask = _apply_view_dropout(teacher_mask, teacher_view_dropout_prob)

            with torch.amp.autocast(device_type=amp_device_type, enabled=amp_active):
                student_logits = model.forward_student(xb)
                teacher_logits = model.forward_teacher(teacher_imgs, teacher_mask, teacher_view_ids_tensor)
                student_loss = _classification_loss(
                    student_logits,
                    yb,
                    criterion,
                    genus_target_matrix=genus_target_matrix,
                )
                teacher_loss = _classification_loss(
                    teacher_logits,
                    yb,
                    criterion,
                    genus_target_matrix=genus_target_matrix,
                )
                teacher_probs = torch.softmax(teacher_logits.detach() / distill_temperature, dim=1)
                student_log_probs = torch.log_softmax(student_logits / distill_temperature, dim=1)
                distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (distill_temperature ** 2)
                loss = student_loss + (teacher_loss_weight * teacher_loss) + (distill_loss_weight * distill_loss)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            if step_i % accum_steps == 0 or step_i == len(train_loader):
                if grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            pred = student_logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            if genus_ids is not None:
                correct_genus += int((genus_ids[pred] == genus_ids[yb]).sum().item())
            total += int(yb.numel())
            running_total += float(loss.item()) * accum_steps
            running_student += float(student_loss.item())
            running_teacher += float(teacher_loss.item())
            running_distill += float(distill_loss.item())
            pbar.set_postfix(
                loss=running_total / max(1, pbar.n + 1),
                student=running_student / max(1, pbar.n + 1),
                teacher=running_teacher / max(1, pbar.n + 1),
                distill=running_distill / max(1, pbar.n + 1),
            )
            del (
                xb,
                yb,
                teacher_imgs,
                teacher_mask,
                teacher_view_ids_tensor,
                student_logits,
                teacher_logits,
                student_loss,
                teacher_loss,
                distill_loss,
                loss,
                pred,
            )
            if device.type == "mps" and (step_i % accum_steps == 0):
                torch.mps.empty_cache()

        acc = eval_acc()
        genus_acc = eval_genus_acc()
        lr_now = _optimizer_lr(optimizer)
        epoch_s = perf_counter() - t0
        if scheduler is not None:
            scheduler.step()

        train_loss = running_total / max(1, len(train_loader))
        train_student_loss = running_student / max(1, len(train_loader))
        train_teacher_loss = running_teacher / max(1, len(train_loader))
        train_distill_loss = running_distill / max(1, len(train_loader))
        train_acc = correct / max(1, total)
        train_genus_acc = (correct_genus / max(1, total)) if genus_ids is not None else float("nan")

        metrics_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_student_loss": float(train_student_loss),
                "train_teacher_loss": float(train_teacher_loss),
                "train_distill_loss": float(train_distill_loss),
                "train_acc": float(train_acc),
                "train_genus_acc": float(train_genus_acc),
                "val_acc": float(acc),
                "val_genus_acc": float(genus_acc),
                "lr": float(lr_now),
                "epoch_seconds": float(epoch_s),
            }
        )
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_student_loss={train_student_loss:.4f} "
            f"train_teacher_loss={train_teacher_loss:.4f} train_distill_loss={train_distill_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={acc:.4f} lr={lr_now:.6g}"
        )

        if save_each_epoch:
            # Written before the early-stopping break, or the stopping epoch is never saved
            # and the history is one checkpoint short of the metrics log.
            torch.save(
                {
                    "model": model.student_model.state_dict(),
                    "backbone": backbone,
                    "input_size": input_size,
                    "epoch": int(epoch),
                    "training_mode": TRAINING_MODE,
                },
                history_dir / f"classifier_epoch{epoch:03d}.pt",
            )

        if val_loader is not None and acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.student_model.state_dict(),
                    "backbone": backbone,
                    "input_size": input_size,
                    "training_mode": TRAINING_MODE,
                },
                ckpt_path,
            )
            torch.save(
                {
                    "teacher_fusion": model.teacher_fusion.state_dict(),
                    "teacher_head": model.teacher_head.state_dict(),
                    "teacher_norm": model.teacher_norm.state_dict(),
                    "view_embed": model.view_embed.state_dict(),
                    "teacher_view_ids": train_ds.teacher_view_ids,
                    "source_priority": [str(s) for s in source_priority],
                    "distill_temperature": float(distill_temperature),
                },
                aux_path,
            )
        elif val_loader is not None and early_stopping_patience > 0:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {early_stopping_patience} epochs).")
                break

    if val_loader is None or not ckpt_path.exists():
        torch.save(
            {
                "model": model.student_model.state_dict(),
                "backbone": backbone,
                "input_size": input_size,
                "training_mode": TRAINING_MODE,
            },
            ckpt_path,
        )
        torch.save(
            {
                "teacher_fusion": model.teacher_fusion.state_dict(),
                "teacher_head": model.teacher_head.state_dict(),
                "teacher_norm": model.teacher_norm.state_dict(),
                "view_embed": model.view_embed.state_dict(),
                "teacher_view_ids": train_ds.teacher_view_ids,
                "source_priority": [str(s) for s in source_priority],
                "distill_temperature": float(distill_temperature),
            },
            aux_path,
        )

    label_map_path = checkpoints_dir / "label_map.json"
    label_map_path.write_text(json.dumps(label_space.idx_to_species, indent=2), encoding="utf-8")
    meta_path = checkpoints_dir / "experiment_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "training_mode": TRAINING_MODE,
                "teacher_view_ids": train_ds.teacher_view_ids,
                "source_priority": [str(s) for s in source_priority],
                "training_metrics": str(metrics_path),
                "config_path": str(Path(config_path).resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainOutput(checkpoint_path=ckpt_path, label_map_path=label_map_path)


if __name__ == "__main__":
    out = train_from_config()
    print(f"checkpoint: {out.checkpoint_path}")
    print(f"labels: {out.label_map_path}")
