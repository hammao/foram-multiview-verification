from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_species_list(species_csv: str | Path) -> list[str]:
    df = pd.read_csv(species_csv)
    if "species" not in df.columns:
        raise ValueError(f"{species_csv} must have a 'species' column.")
    return df["species"].dropna().astype(str).tolist()


def load_species_table(species_csv: str | Path) -> pd.DataFrame:
    """
    Load a species table.

    Required columns:
    - species: display name / class label (may contain spaces)

    Optional columns:
    - slug: folder-safe name used under raw/species/ (recommended; no spaces)
    """
    df = pd.read_csv(species_csv)
    if "species" not in df.columns:
        raise ValueError(f"{species_csv} must have a 'species' column.")
    if "slug" not in df.columns:
        # Backwards compatibility: derive a stable slug.
        df = df.copy()
        df["slug"] = df["species"].astype(str).str.replace(" ", "_", regex=False)
    return df[["species", "slug"]].dropna().astype(str)

