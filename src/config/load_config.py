from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config at {p} must be a mapping (dict).")
    return data


def resolve_path(base_dir: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    checkpoints_dir: Path
    species_csv: Path


@dataclass(frozen=True)
class Config:
    base_dir: Path
    raw: dict[str, Any]
    paths: Paths


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load config YAML as a dict (kept flexible for quick iteration)."""
    cfg_path = Path(config_path).resolve()
    cfg = load_yaml(cfg_path)

    if "paths" not in cfg or not isinstance(cfg["paths"], dict):
        raise ValueError("Config must contain a 'paths' mapping.")

    paths = cfg["paths"]
    # Determine project root robustly:
    # - Historically configs lived directly under <repo>/configs/*.yaml, so cfg_path.parent.parent worked.
    # - We now also support nested config folders like <repo>/configs/ablations/*.yaml.
    base_dir = cfg_path.parent.parent
    for p in [cfg_path.parent, *cfg_path.parents]:
        if p.name == "configs":
            base_dir = p.parent
            break

    processed_dir = resolve_path(base_dir, paths.get("processed_dir", "processed"))
    splits_dir = (
        resolve_path(base_dir, paths["splits_dir"])
        if paths.get("splits_dir")
        else processed_dir / "manifest" / "splits"
    )
    cfg["paths_resolved"] = {
        # Exposed so callers can resolve manifest paths that were written relative to the
        # project root, which is how the public verification repository ships them.
        "base_dir": str(base_dir),
        "raw_dir": str(resolve_path(base_dir, paths.get("raw_dir", "."))),
        "processed_dir": str(processed_dir),
        "checkpoints_dir": str(resolve_path(base_dir, paths.get("checkpoints_dir", "checkpoints"))),
        "species_csv": str(resolve_path(base_dir, paths.get("species_csv", "configs/species.csv"))),
        "splits_dir": str(splits_dir),
    }

    return cfg

