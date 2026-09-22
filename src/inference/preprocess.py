from __future__ import annotations

from PIL import Image, ImageOps


def _letterbox(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    return ImageOps.pad(im, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0))


def _autocrop_otsu_bbox(
    im: Image.Image,
    *,
    margin_frac: float = 0.06,
    min_area_frac: float = 0.01,
) -> Image.Image:
    # Keep inference dependency-light; mirror the training resize autocrop behavior.
    import numpy as np

    rgb = im.convert("RGB")
    g = np.array(rgb.convert("L"), dtype=np.uint8)
    h, w = g.shape
    if h < 4 or w < 4:
        return rgb

    hist = np.bincount(g.reshape(-1), minlength=256).astype(np.float64)
    total = float(g.size)
    if total <= 0:
        return rgb
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b2))

    m_hi = g > t
    m_lo = g < t

    def _bbox(mask):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return x0, y0, x1, y1

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

