"""Rarity plot inputs and the Hoeglundina elegans held-out-view / LOSO probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-report", required=True, help="eval_report_*.json with sklearn report")
    ap.add_argument("--train-csv", default="processed/manifest/splits/train.csv")
    ap.add_argument("--test-csv", default="processed/manifest/splits/test.csv")
    ap.add_argument("--out-dir", default="processed/manifest/probes")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    report = json.loads(Path(args.eval_report).read_text(encoding="utf-8"))
    per = report.get("report") or {}
    train = pd.read_csv(root / args.train_csv if not Path(args.train_csv).is_absolute() else args.train_csv)
    test = pd.read_csv(root / args.test_csv if not Path(args.test_csv).is_absolute() else args.test_csv)
    train_n = train.groupby("species").size().rename("n_train")
    rows = []
    for species, metrics in per.items():
        if species in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "species": species,
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1-score"),
                "support": metrics.get("support"),
                "n_train": int(train_n.get(species, 0)),
            }
        )
    rarity = pd.DataFrame(rows).sort_values("n_train")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rarity.to_csv(out_dir / "rarity_f1_vs_ntrain.csv", index=False)

    he = test[test["species"] == "Hoeglundina elegans"].copy()
    he_metrics = per.get("Hoeglundina elegans") or {}
    specimens = sorted(he["specimen_id"].dropna().unique().tolist())
    probe = {
        "species": "Hoeglundina elegans",
        "n_specimens_in_dataset": 2,
        "n_test_images": int(len(he)),
        "test_images_by_specimen": he.groupby("specimen_id").size().to_dict() if not he.empty else {},
        "held_out_view_metrics": he_metrics,
        "loso_note": (
            "A true leave-one-specimen-out run trains twice, each time dropping one of "
            "the two H. elegans specimens, then evaluates the held-out specimen. Use "
            "scripts/run_helegans_loso.py after the headline models finish. The "
            "held-out-view numbers above are not LOSO: both specimens contribute views "
            "to training."
        ),
        "spearman_f1_vs_ntrain": None,
    }
    if len(rarity) >= 3:
        probe["spearman_f1_vs_ntrain"] = float(rarity["f1"].corr(rarity["n_train"], method="spearman"))
    (out_dir / "hoeglundina_elegans_probe.json").write_text(json.dumps(probe, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"n_species": int(len(rarity)), "spearman_f1_vs_ntrain": probe["spearman_f1_vs_ntrain"], "helegans": he_metrics}, indent=2))


if __name__ == "__main__":
    main()
