from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ViewRecord:
    species: str
    specimen_id: int
    view_id: int
    path: str
    source: str  # stacked_tiff | view_folder_stacked_tiff
    z_index: int | None = None


STACKED_RE = re.compile(r"^(?P<specimen>\d+)-(?P<view>\d+)-stacked-.*-view-(?P<view2>\d+)\.tif$", re.I)
PREFIX_RE = re.compile(r"^(?P<specimen>\d+)-(?P<view>\d+)-")
V2_STACKED_RE = re.compile(r"^S(?P<specimen>\d{3})_A(?P<angle>\d{3})_stacked\.tif$", re.I)
V2_ZSTACK_RE = re.compile(r"^S(?P<specimen>\d{3})_A(?P<angle>\d{3})_Z(?P<z>\d{4})_(?P<rest>.+)\.tif$", re.I)


def _find_stacked_dirs(species_dir: Path) -> list[Path]:
    """Return possible stacked TIFF directories under a species folder."""
    dirs: list[Path] = []
    for p in species_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == "stacked_tiffs":
            dirs.append(p)
        elif re.search(r"^stacked\s*tiffs", p.name, re.I):
            dirs.append(p)
    # Prefer canonical first
    dirs.sort(key=lambda d: (d.name != "stacked_tiffs", d.name))
    return dirs


def _parse_specimen_view_from_filename(name: str) -> tuple[int, int] | None:
    m = STACKED_RE.match(name)
    if m:
        specimen = int(m.group("specimen"))
        view = int(m.group("view"))
        # view2 is sometimes redundant; trust prefix view
        return specimen, view
    m2 = PREFIX_RE.match(name)
    if m2:
        return int(m2.group("specimen")), int(m2.group("view"))
    return None


def _angle_to_view_id(angle_deg: int) -> int:
    # view-1 = 0° (locked-in); allow only multiples of 60
    if angle_deg % 60 != 0:
        raise ValueError(f"Angle must be multiple of 60: {angle_deg}")
    return int(angle_deg // 60) + 1


def _discover_v2_stacked(species_dir: Path, *, species_label: str) -> list[ViewRecord]:
    rows: list[ViewRecord] = []
    # raw/species/<slug>/specimens/S###/angles/A###/stacked/S###_A###_stacked.tif
    specimens_dir = species_dir / "specimens"
    if not specimens_dir.exists():
        return rows

    for tif in sorted(specimens_dir.glob("S*/angles/A*/stacked/*.tif")):
        m = V2_STACKED_RE.match(tif.name)
        if not m:
            continue
        specimen_id = int(m.group("specimen"))
        angle = int(m.group("angle"))
        try:
            view_id = _angle_to_view_id(angle)
        except ValueError:
            continue
        if view_id < 1 or view_id > 6:
            continue
        rows.append(
            ViewRecord(
                species=species_label,
                specimen_id=specimen_id,
                view_id=view_id,
                path=str(tif.resolve()),
                source="stacked_tiff",
                z_index=None,
            )
        )
    return rows


def _sample_evenly_sorted(items: list[Path], *, k: int) -> list[Path]:
    """Pick k items spread roughly evenly across the list (deterministic)."""
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    # Equivalent to uniform sampling on index range.
    idxs = [int(round(i * (len(items) - 1) / float(k - 1))) for i in range(k)]
    out: list[Path] = []
    seen: set[int] = set()
    for j in idxs:
        jj = int(max(0, min(len(items) - 1, j)))
        if jj in seen:
            continue
        seen.add(jj)
        out.append(items[jj])
    # If rounding produced duplicates, fill in remaining from the middle-out.
    if len(out) < k:
        for p in items:
            if len(out) >= k:
                break
            if p not in out:
                out.append(p)
    return out[:k]


def _discover_v2_zstack(
    species_dir: Path,
    *,
    species_label: str,
    stride: int = 1,
    max_per_angle: int | None = None,
) -> list[ViewRecord]:
    """
    Discover per-slice Z-stack TIFFs.

    Canonical location:
      raw/species/<slug>/specimens/S###/angles/A###/zstack/S###_A###_Z####_C00.tif
    """
    rows: list[ViewRecord] = []
    specimens_dir = species_dir / "specimens"
    if not specimens_dir.exists():
        return rows

    stride = max(1, int(stride))
    max_per_angle = int(max_per_angle) if max_per_angle is not None else None
    if max_per_angle is not None and max_per_angle <= 0:
        return rows

    # Group by (specimen, angle) so we can optionally downsample per angle deterministically.
    by_key: dict[tuple[int, int], list[tuple[int, Path]]] = {}
    for tif in sorted(specimens_dir.glob("S*/angles/A*/zstack/*.tif")):
        m = V2_ZSTACK_RE.match(tif.name)
        if not m:
            continue
        specimen_id = int(m.group("specimen"))
        angle = int(m.group("angle"))
        z = int(m.group("z"))
        try:
            _ = _angle_to_view_id(angle)
        except ValueError:
            continue
        by_key.setdefault((specimen_id, angle), []).append((z, tif))

    for (specimen_id, angle), pairs in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        pairs_sorted = sorted(pairs, key=lambda t: t[0])
        # Optional stride
        if stride > 1:
            pairs_sorted = pairs_sorted[::stride]
        # Optional max-per-angle cap (still deterministic)
        if max_per_angle is not None and len(pairs_sorted) > max_per_angle:
            chosen = _sample_evenly_sorted([p for _, p in pairs_sorted], k=max_per_angle)
            chosen_set = {c.resolve() for c in chosen}
            pairs_sorted = [(z, p) for z, p in pairs_sorted if p.resolve() in chosen_set]

        try:
            view_id = _angle_to_view_id(angle)
        except ValueError:
            continue
        if view_id < 1 or view_id > 6:
            continue

        for z, tif in pairs_sorted:
            rows.append(
                ViewRecord(
                    species=species_label,
                    specimen_id=int(specimen_id),
                    view_id=int(view_id),
                    path=str(tif.resolve()),
                    source="zstack_slice",
                    z_index=int(z),
                )
            )

    return rows


def discover_views(raw_dir: Path, species_table: pd.DataFrame) -> list[ViewRecord]:
    rows: list[ViewRecord] = []
    for _, r in species_table.iterrows():
        species = str(r["species"])
        slug = str(r.get("slug", species)).strip()
        species_dir = raw_dir / slug
        if not species_dir.exists():
            continue

        # v2 preferred
        rows.extend(_discover_v2_stacked(species_dir, species_label=species))

        # 1) Prefer stacked_tiffs directories
        for stacked_dir in _find_stacked_dirs(species_dir):
            for tif in sorted(stacked_dir.glob("*.tif")):
                parsed = _parse_specimen_view_from_filename(tif.name)
                if not parsed:
                    continue
                specimen_id, view_id = parsed
                rows.append(
                    ViewRecord(
                        species=species,
                        specimen_id=specimen_id,
                        view_id=view_id,
                        path=str(tif.resolve()),
                        source="stacked_tiff",
                        z_index=None,
                    )
                )

        # 2) Also include stacked files sitting inside view folders (some species have them)
        for vd in sorted(species_dir.iterdir()):
            if not vd.is_dir():
                continue
            # Matches patterns like "5-1-Adelosina pulchella-view-1"
            if not re.match(r"^\d+-\d+-.*-view-\d+$", vd.name):
                continue
            for tif in sorted(vd.glob("*-stacked-*.tif")):
                parsed = _parse_specimen_view_from_filename(tif.name)
                if not parsed:
                    continue
                specimen_id, view_id = parsed
                rows.append(
                    ViewRecord(
                        species=species,
                        specimen_id=specimen_id,
                        view_id=view_id,
                        path=str(tif.resolve()),
                        source="view_folder_stacked_tiff",
                        z_index=None,
                    )
                )

    return rows


def write_manifest_csv(records: list[ViewRecord], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        df = pd.DataFrame(columns=["species", "specimen_id", "view_id", "path", "source", "z_index"])
    # Ensure stable column presence even if some sources omit z_index.
    if "z_index" not in df.columns:
        df["z_index"] = None
    df = df.sort_values(["species", "specimen_id", "view_id", "source", "z_index"], ignore_index=True)
    df.to_csv(out_csv, index=False)


def discover_from_config(config: dict) -> Path:
    raw_dir = Path(config["paths_resolved"]["raw_dir"])
    species_csv = Path(config["paths_resolved"]["species_csv"])
    processed_dir = Path(config["paths_resolved"]["processed_dir"])

    species_table = pd.read_csv(species_csv)
    if "slug" not in species_table.columns:
        species_table = species_table.copy()
        species_table["slug"] = species_table["species"].astype(str).str.replace(" ", "_", regex=False)
    records = discover_views(raw_dir=raw_dir, species_table=species_table[["species", "slug"]])
    out_csv = processed_dir / "manifest" / "views_stacked.csv"
    write_manifest_csv(records, out_csv)
    return out_csv


def discover_zstack_from_config(config: dict) -> Path:
    raw_dir = Path(config["paths_resolved"]["raw_dir"])
    species_csv = Path(config["paths_resolved"]["species_csv"])
    processed_dir = Path(config["paths_resolved"]["processed_dir"])

    species_table = pd.read_csv(species_csv)
    if "slug" not in species_table.columns:
        species_table = species_table.copy()
        species_table["slug"] = species_table["species"].astype(str).str.replace(" ", "_", regex=False)

    z_cfg = config.get("data", {}).get("zstack", {}) if isinstance(config.get("data", {}), dict) else {}
    if not isinstance(z_cfg, dict):
        z_cfg = {}
    include = bool(z_cfg.get("include", True))
    stride = int(z_cfg.get("stride", 1))
    max_per_angle = z_cfg.get("max_per_angle", None)
    max_per_angle = int(max_per_angle) if max_per_angle is not None else None

    records: list[ViewRecord] = []
    if include:
        for _, r in species_table.iterrows():
            species = str(r["species"])
            slug = str(r.get("slug", species)).strip()
            species_dir = raw_dir / slug
            if not species_dir.exists():
                continue
            records.extend(
                _discover_v2_zstack(
                    species_dir,
                    species_label=species,
                    stride=stride,
                    max_per_angle=max_per_angle,
                )
            )

    out_csv = processed_dir / "manifest" / "views_zstack.csv"
    write_manifest_csv(records, out_csv)
    return out_csv

