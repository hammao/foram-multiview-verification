from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class GifFrameRecord:
    species: str
    specimen_id: int
    view_id: int
    path: str
    source: str  # gif_original | gif_rotated90


def _sample_frame_indices(n_frames: int, target: int = 6) -> list[int]:
    if n_frames <= 0:
        return []
    if n_frames == target:
        return list(range(n_frames))
    # Uniformly sample target frames from [0, n_frames-1]
    return [int(round(i * (n_frames - 1) / (target - 1))) for i in range(target)]


def _frame_to_view_id(frame_index: int, mapping: list[int] | None) -> int:
    if mapping is None:
        return frame_index + 1
    if len(mapping) != 6:
        raise ValueError("gif_frame_to_view mapping must be length 6.")
    return int(mapping[frame_index])


def _extract_one_gif(
    gif_path: Path,
    out_dir: Path,
    species: str,
    specimen_id: int,
    source: str,
    expected_frames: int,
    mapping: list[int] | None,
) -> list[GifFrameRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(gif_path)

    # Count frames
    n = getattr(img, "n_frames", 1)
    indices = _sample_frame_indices(n_frames=n, target=expected_frames)

    records: list[GifFrameRecord] = []
    for i, frame_idx in enumerate(indices):
        img.seek(frame_idx)
        frame = img.convert("RGB")
        view_id = _frame_to_view_id(i, mapping=mapping)
        out_name = f"{species}_{specimen_id}_view{view_id}_{source}.png"
        out_path = out_dir / out_name
        # Avoid re-writing frames on repeated pipeline runs (huge speedup on large datasets).
        if not out_path.exists():
            frame.save(out_path)
        records.append(
            GifFrameRecord(
                species=species,
                specimen_id=specimen_id,
                view_id=view_id,
                path=str(out_path.resolve()),
                source=source,
            )
        )

    return records


def extract_gifs_from_species_dir(
    species_dir: Path,
    *,
    species_label: str | None = None,
    processed_dir: Path,
    gif_expected_frames: int = 6,
    gif_frame_to_view: list[int] | None = None,
    include_rotated90: bool = True,
) -> list[GifFrameRecord]:
    records: list[GifFrameRecord] = []

    label = species_label or species_dir.name

    # v2: raw/species/<slug>/specimens/S###/gifs/original.gif
    specimens_dir = species_dir / "specimens"
    if specimens_dir.exists():
        for sd in sorted([p for p in specimens_dir.iterdir() if p.is_dir()]):
            if not sd.name.startswith("S") and sd.name != "UNASSIGNED":
                continue
            if sd.name == "UNASSIGNED":
                specimen_id = 0
            else:
                try:
                    specimen_id = int(sd.name[1:])
                except ValueError:
                    specimen_id = 0

            original = sd / "gifs" / "original.gif"
            rotated = sd / "gifs" / "rotated_90.gif"

            out_dir = processed_dir / "gif_frames" / label
            if original.exists():
                records.extend(
                    _extract_one_gif(
                        gif_path=original,
                        out_dir=out_dir,
                        species=label,
                        specimen_id=specimen_id,
                        source="gif_original",
                        expected_frames=gif_expected_frames,
                        mapping=gif_frame_to_view,
                    )
                )
            if include_rotated90 and rotated.exists():
                records.extend(
                    _extract_one_gif(
                        gif_path=rotated,
                        out_dir=out_dir,
                        species=label,
                        specimen_id=specimen_id,
                        source="gif_rotated90",
                        expected_frames=gif_expected_frames,
                        mapping=gif_frame_to_view,
                    )
                )

        return records

    # legacy fallback: GIFs at species root
    # Heuristic: specimen_id is the first number that appears in stacked tif filenames if present.
    specimen_id = None
    for p in (species_dir / "stacked_tiffs").glob("*.tif"):
        m = None
        try:
            m = p.name.split("-", 2)
        except Exception:  # noqa: BLE001
            m = None
        if m and len(m) >= 2 and m[0].isdigit():
            specimen_id = int(m[0])
            break
    if specimen_id is None:
        specimen_id = 0

    original = species_dir / "original.gif"
    rotated = species_dir / "rotated_90.gif"

    out_dir = processed_dir / "gif_frames" / label
    if original.exists():
        records.extend(
            _extract_one_gif(
                gif_path=original,
                out_dir=out_dir,
                species=label,
                specimen_id=specimen_id,
                source="gif_original",
                expected_frames=gif_expected_frames,
                mapping=gif_frame_to_view,
            )
        )
    if include_rotated90 and rotated.exists():
        records.extend(
            _extract_one_gif(
                gif_path=rotated,
                out_dir=out_dir,
                species=label,
                specimen_id=specimen_id,
                source="gif_rotated90",
                expected_frames=gif_expected_frames,
                mapping=gif_frame_to_view,
            )
        )

    return records


def write_gif_manifest(records: list[GifFrameRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        df = pd.DataFrame(columns=["species", "specimen_id", "view_id", "path", "source"])
    df = df.sort_values(["species", "specimen_id", "view_id", "source"], ignore_index=True)
    df.to_csv(out_csv, index=False)

