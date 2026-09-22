from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from src.classification.dataset import LabelSpace


def _safe_int(value) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def _load_image(path: str | Path) -> Image.Image:
    return Image.open(Path(str(path))).convert("RGB")


def _apply_transform(img: Image.Image, transform):
    if transform is None:
        return pil_to_tensor(img).float() / 255.0
    return transform(img)


class MultiViewDistillationDataset(Dataset):
    """
    Each row is still a single-image student sample, but we also attach a deterministic
    multi-view specimen context for the teacher branch.

    Teacher context selection is intentionally conservative:
    - one representative image per available rotational view
    - source priority: stacked TIFF > original GIF frame > median Z-slice
    - only images already present in the train split are used, so holdout views remain held out
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        label_space: LabelSpace,
        image_col: str = "path_resized",
        student_transform=None,
        teacher_transform=None,
        teacher_view_ids: list[int] | None = None,
        source_priority: list[str] | None = None,
        zstack_selection: str = "median",
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.label_space = label_space
        self.image_col = image_col
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform
        self.source_priority = [str(s) for s in (source_priority or ["stacked_tiff", "gif_original", "zstack_slice"])]
        self.zstack_selection = str(zstack_selection).lower().strip() or "median"

        if self.df.empty:
            raise ValueError(f"Dataset CSV is empty: {csv_path}")
        if "species" not in self.df.columns:
            raise ValueError("CSV must contain 'species' column.")
        if image_col not in self.df.columns:
            raise ValueError(f"CSV must contain '{image_col}' column.")
        if "specimen_id" not in self.df.columns:
            raise ValueError("CSV must contain 'specimen_id' column for multi-view distillation.")
        if "view_id" not in self.df.columns:
            raise ValueError("CSV must contain 'view_id' column for multi-view distillation.")

        self.teacher_view_ids = (
            [int(v) for v in teacher_view_ids]
            if teacher_view_ids
            else self._discover_teacher_view_ids()
        )
        if not self.teacher_view_ids:
            raise ValueError("Could not determine teacher view ids from the split CSV.")

        self.teacher_contexts = self._build_teacher_contexts()

    def _discover_teacher_view_ids(self) -> list[int]:
        view_ids = []
        for value in self.df["view_id"].tolist():
            v = _safe_int(value)
            if v is not None:
                view_ids.append(v)
        return sorted(set(view_ids))

    def _pick_row_for_source(self, group: pd.DataFrame, source_name: str) -> pd.Series | None:
        if "source" not in group.columns:
            return None
        g = group.loc[group["source"].astype(str) == str(source_name)].copy()
        if g.empty:
            return None
        if str(source_name) == "zstack_slice" and self.zstack_selection == "median" and "z_index" in g.columns:
            g["__z_index"] = pd.to_numeric(g["z_index"], errors="coerce")
            g = g.sort_values(["__z_index", self.image_col], na_position="last").reset_index(drop=True)
            return g.iloc[len(g) // 2]
        g = g.sort_values(self.image_col).reset_index(drop=True)
        return g.iloc[0]

    def _pick_representative_row(self, group: pd.DataFrame) -> pd.Series:
        for source_name in self.source_priority:
            row = self._pick_row_for_source(group, source_name)
            if row is not None:
                return row
        return group.sort_values(self.image_col).reset_index(drop=True).iloc[0]

    def _build_teacher_contexts(self) -> dict[tuple[str, str], dict[int, str]]:
        contexts: dict[tuple[str, str], dict[int, str]] = {}
        group_cols = ["species", "specimen_id", "view_id"]
        for (species, specimen_id, view_id), group in self.df.groupby(group_cols, sort=False):
            view_id_i = _safe_int(view_id)
            if view_id_i is None:
                continue
            picked = self._pick_representative_row(group)
            key = (str(species), str(specimen_id))
            contexts.setdefault(key, {})[view_id_i] = str(picked[self.image_col])
        return contexts

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(str(row[self.image_col]))
        species = str(row["species"])
        specimen_id = str(row["specimen_id"])
        y = self.label_space.species_to_idx[species]

        student_img = _apply_transform(_load_image(img_path), self.student_transform)
        empty_teacher = torch.zeros_like(student_img)

        teacher_imgs: list[torch.Tensor] = []
        teacher_mask: list[bool] = []
        teacher_context = self.teacher_contexts.get((species, specimen_id), {})
        for view_id in self.teacher_view_ids:
            teacher_path = teacher_context.get(int(view_id))
            if teacher_path:
                teacher_img = _apply_transform(_load_image(teacher_path), self.teacher_transform)
                teacher_imgs.append(teacher_img)
                teacher_mask.append(True)
            else:
                teacher_imgs.append(empty_teacher.clone())
                teacher_mask.append(False)

        return (
            student_img,
            torch.tensor(y, dtype=torch.long),
            torch.stack(teacher_imgs, dim=0),
            torch.tensor(teacher_mask, dtype=torch.bool),
            torch.tensor(self.teacher_view_ids, dtype=torch.long),
        )
