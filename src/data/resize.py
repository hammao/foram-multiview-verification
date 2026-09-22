from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageFile

# Be tolerant to partially-written PNGs/TIFFs from acquisition pipelines.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ResizeConfig:
    input_size: int
    format: str = "png"
    # Optional preprocessing crop to reduce background variability.
    # - "none": no crop
    # - "otsu_bbox": Otsu threshold -> pick smaller foreground/background mask -> bbox crop
    autocrop: str = "none"
    autocrop_margin_frac: float = 0.06
    autocrop_min_area_frac: float = 0.01
    overwrite: bool = False


def _letterbox(im: Image.Image, size: int) -> Image.Image:
    """Resize preserving aspect ratio and pad to square."""
    im = im.convert("RGB")
    return ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def _otsu_threshold(gray_u8):
    import numpy as np

    hist = np.bincount(gray_u8.reshape(-1), minlength=256).astype(np.float64)
    total = gray_u8.size
    if total <= 0:
        return 0
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    # Between-class variance; avoid divide-by-zero.
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b2))
    return t


def _autocrop_otsu_bbox(
    im: Image.Image,
    *,
    margin_frac: float = 0.06,
    min_area_frac: float = 0.01,
) -> Image.Image:
    """
    Crop to a bounding box around the specimen using a simple Otsu-threshold heuristic.
    Works best when the specimen occupies a minority of pixels (common in microscope imagery).
    """
    import numpy as np

    rgb = im.convert("RGB")
    g = np.array(rgb.convert("L"), dtype=np.uint8)
    h, w = g.shape
    if h < 4 or w < 4:
        return rgb

    t = _otsu_threshold(g)
    m_hi = g > t
    m_lo = g < t
    total = float(h * w)

    def _bbox(mask):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return x0, y0, x1, y1

    # Choose the smaller "foreground" mask that still has reasonable area.
    cand = []
    for m in (m_hi, m_lo):
        area = float(np.count_nonzero(m))
        if area / total < float(min_area_frac):
            continue
        if area / total > 0.95:
            continue
        bb = _bbox(m)
        if bb is not None:
            cand.append((area, bb))
    if not cand:
        return rgb

    _, (x0, y0, x1, y1) = sorted(cand, key=lambda z: z[0])[0]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    mx = int(bw * float(margin_frac)) + 2
    my = int(bh * float(margin_frac)) + 2
    x0 = max(0, x0 - mx)
    y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx)
    y1 = min(h, y1 + my)
    if (x1 - x0) < 4 or (y1 - y0) < 4:
        return rgb
    return rgb.crop((x0, y0, x1, y1))


def resize_manifest(manifest_csv: Path, out_dir: Path, cfg: ResizeConfig) -> Path:
    df = pd.read_csv(manifest_csv)
    if df.empty:
        out_csv = out_dir.parent / "manifest" / "views_resized.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        return out_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[str] = []
    n_errors = 0

    for _, row in df.iterrows():
        src = Path(str(row["path"]))
        species = str(row["species"])
        specimen_id = int(row["specimen_id"])
        view_id = int(row["view_id"])
        source = str(row.get("source", "unknown"))
        z_index = None
        if "z_index" in df.columns:
            try:
                zv = row.get("z_index", None)
                if zv is not None and str(zv) != "" and str(zv).lower() != "nan":
                    z_index = int(zv)
            except Exception:
                z_index = None

        # Important: zstack slices need a unique name per Z index to avoid collisions.
        if z_index is None:
            out_name = f"{species}_{specimen_id}_view{view_id}_{source}.{cfg.format}"
        else:
            out_name = f"{species}_{specimen_id}_view{view_id}_{source}_z{int(z_index):04d}.{cfg.format}"
        dst = out_dir / out_name

        if cfg.overwrite or (not dst.exists()):
            try:
                im = Image.open(src)
                # Force decoding now so truncation issues throw here.
                im.load()
                if str(cfg.autocrop).lower().strip() == "otsu_bbox":
                    im = _autocrop_otsu_bbox(
                        im,
                        margin_frac=float(cfg.autocrop_margin_frac),
                        min_area_frac=float(cfg.autocrop_min_area_frac),
                    )
                resized = _letterbox(im, cfg.input_size)
                resized.save(dst)
            except Exception as e:  # noqa: BLE001
                n_errors += 1
                # If an older resized file exists, keep it; otherwise write a placeholder so downstream steps run.
                if not dst.exists():
                    placeholder = Image.new("RGB", (cfg.input_size, cfg.input_size), color=(0, 0, 0))
                    placeholder.save(dst)
                print(f"[resize] WARN: failed to process {src}: {e!r}")

        out_paths.append(str(dst.resolve()))

    df["path_resized"] = out_paths
    out_csv = out_dir.parent / "manifest" / "views_resized.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if n_errors:
        print(f"[resize] Completed with {n_errors} unreadable images (kept existing or wrote placeholders).")
    return out_csv

