from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config.load_config import load_config


def run_yolov11_crops(manifest_csv: Path, out_dir: Path, checkpoint: str, conf: float, iou: float) -> Path:
    try:
        from ultralytics import YOLO
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("ultralytics is not installed; pip install ultralytics") from e

    model = YOLO(checkpoint)
    df = pd.read_csv(manifest_csv)
    if df.empty:
        return manifest_csv

    out_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[str] = []

    for _, row in df.iterrows():
        img_path = Path(str(row.get("path_resized") or row["path"]))
        im = Image.open(img_path).convert("RGB")

        results = model.predict(source=str(img_path), conf=conf, iou=iou, verbose=False)
        r = results[0]

        # Pick the highest-confidence box, if any. If none, fall back to full image.
        if r.boxes is None or len(r.boxes) == 0:
            crop = im
        else:
            boxes = r.boxes
            best = boxes.conf.argmax().item()
            xyxy = boxes.xyxy[best].tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            crop = im.crop((x1, y1, x2, y2))

        species = str(row["species"])
        specimen_id = int(row["specimen_id"])
        view_id = int(row["view_id"])
        source = str(row.get("source", "unknown"))
        out_name = f"{species}_{specimen_id}_view{view_id}_{source}_crop.png"
        out_path = out_dir / out_name
        crop.save(out_path)
        crop_paths.append(str(out_path.resolve()))

    df["path_crop"] = crop_paths
    out_csv = manifest_csv.parent / "views_with_crops.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default="processed/manifest/views_resized.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths_resolved"]["processed_dir"])
    out_dir = processed_dir / "crops"

    det = cfg.get("detection", {})
    checkpoint = str(det.get("detector_checkpoint", "yolov11l-seg.pt"))
    conf = float(det.get("conf", 0.25))
    iou = float(det.get("iou", 0.7))

    out = run_yolov11_crops(Path(args.manifest), out_dir=out_dir, checkpoint=checkpoint, conf=conf, iou=iou)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

