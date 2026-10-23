#!/usr/bin/env python3
"""Placeholder Plates 1-3 (hyaline, agglutinated, porcelaneous) for the 50-species paper.

Layout follows manuscript/October_2026_Revisions/Figure_plate_1.png: black background, each
species shown in three rotational views (a = 0 deg, b = 120 deg, c = 240 deg) of its reference
specimen (lowest specimen id, always a training shell), numbered consecutively within the plate.
Images are the 384 px letterboxed EDOF composites from processed/drive140/resized, so the final
plates can be swapped in without changing captions or numbering.

Writes manuscript/October_2026_Revisions/plates/Plate_{1,2,3}_<wall>.png and plate_captions.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TABLE1 = ROOT / "reports" / "experiments" / "2026-10_drive140_paper" / "table1_species_taxonomy.csv"
VIEWS = ROOT / "processed" / "drive140" / "manifest" / "views_resized.csv"
OUT = ROOT / "manuscript" / "October_2026_Revisions" / "plates"
PLATES = [("1", "hyaline"), ("2", "agglutinated"), ("3", "porcelaneous")]
VIEW_IDS = [1, 3, 5]  # 0, 120, 240 degrees
TILE = 300
PAD = 8
COLS = 3  # species per row
LABEL_H = 26


def font(size: int) -> ImageFont.ImageFont:
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick(views: pd.DataFrame, species: str, sid: int, view_id: int) -> Path | None:
    part = views[(views.species == species) & (views.specimen_id == sid) & (views.view_id == view_id)]
    for src in ("stacked_tiff", "pmax_tiff", "gif_original"):
        hit = part[part.source == src]
        if not hit.empty:
            return Path(str(hit.iloc[0]["path_resized"]))
    return None


def main() -> int:
    tax = pd.read_csv(TABLE1)
    views = pd.read_csv(VIEWS)
    OUT.mkdir(parents=True, exist_ok=True)
    f_label = font(22)
    captions = ["# Plates 1-3 (placeholders built from the Drive140 EDOF composites)\n"]
    for number, wall in PLATES:
        rows = tax[tax.wall_type == wall].sort_values(["order", "family", "display_name"]).reset_index(drop=True)
        n = len(rows)
        n_rows = (n + COLS - 1) // COLS
        cell_w = VIEW_IDS.__len__() * (TILE + PAD) + PAD * 2
        width = COLS * cell_w + PAD
        height = n_rows * (TILE + LABEL_H + PAD * 2) + PAD
        canvas = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(canvas)
        cap = [f"\n## Plate {number}. {wall.capitalize()} species ({n} species, three rotational views each: a = 0°, b = 120°, c = 240°)\n"]
        for i, r in rows.iterrows():
            sid = int(str(r.train_specimen_ids).split(";")[0].lstrip("S"))
            row_i, col_i = divmod(i, COLS)
            x0 = PAD + col_i * cell_w
            y0 = PAD + row_i * (TILE + LABEL_H + PAD * 2)
            for k, (vid, letter) in enumerate(zip(VIEW_IDS, "abc", strict=True)):
                p = pick(views, r.code_label, sid, vid)
                x = x0 + PAD + k * (TILE + PAD)
                if p is not None and p.is_file():
                    im = Image.open(p).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
                    canvas.paste(im, (x, y0))
                draw.text((x + 4, y0 + TILE + 2), f"{i + 1}{letter}", fill="white", font=f_label)
            cap.append(f"{i + 1}. *{r.display_name}* (specimen S{sid:03d}; {r.family}, {r.order})")
        out_png = OUT / f"Plate_{number}_{wall}.png"
        canvas.save(out_png, optimize=True)
        cap.append("\nScale bars to be added on the final plate; all tiles are 384 px letterboxed composites at the acquisition zoom of each specimen.")
        captions.extend(cap)
        print(f"Plate {number} {wall}: {n} species -> {out_png.name} ({width}x{height})")
    (OUT / "plate_captions.md").write_text("\n".join(captions) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
