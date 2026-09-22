from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabelSpace:
    species_to_idx: dict[str, int]
    idx_to_species: list[str]


def build_label_space(species: list[str]) -> LabelSpace:
    idx_to_species = list(species)
    species_to_idx = {s: i for i, s in enumerate(idx_to_species)}
    return LabelSpace(species_to_idx=species_to_idx, idx_to_species=idx_to_species)


class ViewDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        label_space: LabelSpace,
        image_col: str = "path_resized",
        transform=None,
        root: str | Path | None = None,
    ):
        """``root`` resolves image paths that the manifest stores relative to a project root.

        Manifests written by the training pipeline hold absolute paths, which are used as-is.
        The public verification repository ships relative ones so the split travels with the
        images.
        """
        self.df = pd.read_csv(csv_path)
        self.label_space = label_space
        self.image_col = image_col
        self.transform = transform
        self.root = Path(root) if root is not None else None

        if self.df.empty:
            raise ValueError(f"Dataset CSV is empty: {csv_path}")
        if "species" not in self.df.columns:
            raise ValueError("CSV must contain 'species' column.")
        if image_col not in self.df.columns:
            raise ValueError(f"CSV must contain '{image_col}' column.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(str(row[self.image_col]))
        if self.root is not None and not img_path.is_absolute():
            img_path = self.root / img_path
        species = str(row["species"])
        y = self.label_space.species_to_idx[species]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)

