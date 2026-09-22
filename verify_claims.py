#!/usr/bin/env python3
"""Recompute every quantitative claim in the manuscript from the released outputs.

No GPU, no model weights and no network access are needed: each check recomputes a number
from the prediction and metric files in ``results/`` and compares it with the value printed
in the paper, which is hard-coded here. A disagreement between the paper and the data is a
FAIL, so this script is a test of the manuscript rather than a summary of it.

    python verify_claims.py            # run everything
    python verify_claims.py --group audit

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

# Species names are italicised in the paper; plain text here.
SINGLE_ERRORS = {
    ("Quinqueloculina pulchella", "Quinqueloculina crassicarinata"): 1,
    ("Spiroloculina communis", "Triloculina plicata"): 1,
}


class Checker:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str, str]] = []

    def check(self, group: str, label: str, actual, expected, places: int = 3) -> None:
        if isinstance(expected, float):
            ok = round(float(actual), places) == round(float(expected), places)
            shown_a, shown_e = f"{float(actual):.{places}f}", f"{float(expected):.{places}f}"
        else:
            ok = actual == expected
            shown_a, shown_e = str(actual), str(expected)
        self.rows.append((group, label, ok, shown_a, shown_e))

    def report(self, only: str | None) -> int:
        rows = [r for r in self.rows if only is None or r[0] == only]
        if not rows:
            print(f"no checks in group {only!r}")
            return 1
        width = max(len(r[1]) for r in rows) + 2
        current = None
        for group, label, ok, actual, expected in rows:
            if group != current:
                print(f"\n{group}")
                current = group
            status = "PASS" if ok else "FAIL"
            detail = f"{actual}" if ok else f"{actual}  (paper says {expected})"
            print(f"  {status}  {label:<{width}} {detail}")
        failed = [r for r in rows if not r[2]]
        print(f"\n{len(rows) - len(failed)}/{len(rows)} checks passed")
        if failed:
            print("FAILED: " + "; ".join(r[1] for r in failed))
        return 1 if failed else 0


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(arm: str, name: str):
    path = RESULTS / arm / name
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return pd.read_csv(path)


def check_dataset(c: Checker) -> None:
    g = "Dataset and splits (Section 3.1)"
    counts = {s: len(pd.read_csv(DATA / "manifest" / "splits" / f"{s}.csv")) for s in ("train", "val", "test")}
    c.check(g, "train images", counts["train"], 5563)
    c.check(g, "validation images", counts["val"], 202)
    c.check(g, "test images", counts["test"], 202)
    c.check(g, "images entering the modelling partitions", sum(counts.values()), 5967)
    c.check(g, "acquired images minus modelled images", 8219 - sum(counts.values()), 2252)

    frames = {s: pd.read_csv(DATA / "manifest" / "splits" / f"{s}.csv") for s in counts}
    species = set(frames["test"]["species"])
    c.check(g, "species in the test split", len(species), 31)

    # No image may appear in two partitions.
    paths = [set(df["path_resized"]) for df in frames.values()]
    overlap = (paths[0] & paths[1]) | (paths[0] & paths[2]) | (paths[1] & paths[2])
    c.check(g, "images shared between partitions", len(overlap), 0)

    # The paper's central caveat: this is a view-level split, so specimens recur across
    # partitions. Verifying it is what makes the internal scores interpretable.
    def specimens(df):
        return set(zip(df["species"], df["specimen_id"]))

    shared = specimens(frames["train"]) & specimens(frames["test"])
    c.check(g, "specimens shared by train and test (view-level split)", len(shared), 32)


def check_test_metrics(c: Checker) -> None:
    g = "Held-out-view test split (Section 3.2, Table 3)"
    expected = {
        "single_stream": (0.990, 0.991, 0.990, 200, 2),
        "dual_stream": (0.916, 0.909, 0.904, 185, 17),
    }
    for arm, (acc, macro, weighted, correct, errors) in expected.items():
        rep = load(arm, "eval_report.json")["report"]
        name = arm.replace("_", "-")
        c.check(g, f"{name} accuracy", rep["accuracy"], acc)
        c.check(g, f"{name} macro F1", rep["macro avg"]["f1-score"], macro)
        c.check(g, f"{name} weighted F1", rep["weighted avg"]["f1-score"], weighted)

        preds = load(arm, "test_predictions.csv")
        c.check(g, f"{name} correct predictions", int(preds["correct"].sum()), correct)
        c.check(g, f"{name} misclassified images", int((preds["correct"] == 0).sum()), errors)
        # Accuracy must follow from the per-image predictions, not only from the report.
        c.check(g, f"{name} accuracy recomputed from predictions", preds["correct"].mean(), acc)

    preds = load("single_stream", "test_predictions.csv")
    wrong = preds[preds["correct"] == 0]
    pairs = dict(zip(zip(wrong["species"], wrong["predicted"]), [1] * len(wrong)))
    c.check(g, "single-stream error pairs match the two named in the caption", pairs, SINGLE_ERRORS)


def check_calibration(c: Checker) -> None:
    g = "Calibration (Section 3.2, Table 3)"
    expected = {
        "single_stream": dict(t=0.546, ece=0.178, ece_cal=0.044, nll=0.230, nll_cal=0.069,
                              val_ece=0.145, val_ece_cal=0.028, val_nll=0.243, val_nll_cal=0.129),
        "dual_stream": dict(t=0.658, ece=0.109, ece_cal=0.049, nll=0.455, nll_cal=0.447,
                            val_ece=0.133, val_ece_cal=0.018, val_nll=0.273, val_nll_cal=0.188),
    }
    for arm, exp in expected.items():
        name = arm.replace("_", "-")
        report = load(arm, "eval_report.json")
        temp = load(arm, "temperature.json")
        c.check(g, f"{name} fitted temperature", temp["temperature"], exp["t"])
        c.check(g, f"{name} temperature fitted on the validation split", temp["split"], "val")
        c.check(g, f"{name} temperature used at evaluation", report["temperature"], exp["t"])
        c.check(g, f"{name} test ECE before scaling", report["ece"], exp["ece"])
        c.check(g, f"{name} test ECE after scaling", report["ece_calibrated"], exp["ece_cal"])
        c.check(g, f"{name} test NLL before scaling", report["nll"], exp["nll"])
        c.check(g, f"{name} test NLL after scaling", report["nll_calibrated"], exp["nll_cal"])
        c.check(g, f"{name} validation ECE before scaling", temp["ece_before"], exp["val_ece"])
        c.check(g, f"{name} validation ECE after scaling", temp["ece_after"], exp["val_ece_cal"])
        c.check(g, f"{name} validation NLL before scaling", temp["nll_before"], exp["val_nll"])
        c.check(g, f"{name} validation NLL after scaling", temp["nll_after"], exp["val_nll_cal"])


def check_audit(c: Checker) -> None:
    g = "Locked Leica audit cohort (Section 3.5)"
    expected = {
        "single_stream": dict(top1=0.422, lo=0.290, hi=0.567, top3=0.778, t1=19, t3=35,
                              normal=0.448, difficult=0.375),
        "dual_stream": dict(top1=0.511, lo=0.370, hi=0.650, top3=0.756, t1=23, t3=34,
                            normal=0.517, difficult=0.500),
    }
    for arm, exp in expected.items():
        name = arm.replace("_", "-")
        s = load(arm, "audit_summary.json")
        c.check(g, f"{name} cohort size", s["n"], 45)
        c.check(g, f"{name} audit cohort excluded from checkpoint selection",
                s["audit_cohort_used_for_selection"], False)
        c.check(g, f"{name} top-1", s["pooled"]["top1"], exp["top1"])
        c.check(g, f"{name} top-3", s["pooled"]["top3"], exp["top3"])
        c.check(g, f"{name} top-1 Wilson lower bound", s["pooled"]["top1_wilson_lo"], exp["lo"])
        c.check(g, f"{name} top-1 Wilson upper bound", s["pooled"]["top1_wilson_hi"], exp["hi"])
        c.check(g, f"{name} normal-stratum top-1 (n=29)", s["normal"]["top1"], exp["normal"])
        c.check(g, f"{name} difficult-stratum top-1 (n=16)", s["difficult"]["top1"], exp["difficult"])

        # Recompute from the per-image predictions rather than trusting the summary.
        preds = load(arm, "audit_predictions.csv")
        top1 = int((preds["pred_1"] == preds["true_species"]).sum())
        top3 = int(
            preds[["pred_1", "pred_2", "pred_3"]].eq(preds["true_species"], axis=0).any(axis=1).sum()
        )
        c.check(g, f"{name} top-1 hits recomputed", top1, exp["t1"])
        c.check(g, f"{name} top-3 hits recomputed", top3, exp["t3"])
        lo, hi = wilson(top1, len(preds))
        c.check(g, f"{name} Wilson interval recomputed", (round(lo, 3), round(hi, 3)),
                (round(exp["lo"], 3), round(exp["hi"], 3)))

    labels = pd.read_csv(DATA / "audit" / "verified_labels_audit_v2.csv")
    c.check(g, "audit label rows", len(labels), 45)
    c.check(g, "normal stratum size", int((labels["difficulty_stratum"] == "normal").sum()), 29)
    c.check(g, "difficult stratum size", int((labels["difficulty_stratum"] == "difficult").sum()), 16)
    c.check(g, "audit images present on disk",
            sum((ROOT / p).exists() for p in labels["image_path"]), 45)


def check_coverage(c: Checker) -> None:
    g = "Coverage-accuracy curve (Figure 6)"
    s = load("single_stream", "audit_confidence_summary.json")
    curve = {round(r["threshold"], 2): r for r in s["coverage_accuracy_curve"]}
    for threshold, covered, accuracy in ((0.2, 45, 0.422), (0.5, 36, 0.528), (0.9, 15, 0.733)):
        row = curve[threshold]
        c.check(g, f"threshold {threshold}: images covered", row["n_covered"], covered)
        c.check(g, f"threshold {threshold}: accuracy on covered", row["accuracy_on_covered"], accuracy)

    ordered = sorted(s["coverage_accuracy_curve"], key=lambda r: r["threshold"])
    accs = [r["accuracy_on_covered"] for r in ordered]
    monotone = all(b >= a - 1e-12 for a, b in zip(accs, accs[1:]))
    c.check(g, "accuracy rises monotonically with the threshold", monotone, True)

    preds = load("single_stream", "audit_confidence_predictions.csv")
    for threshold, covered, accuracy in ((0.2, 45, 0.422), (0.5, 36, 0.528), (0.9, 15, 0.733)):
        sub = preds[preds["confidence"] >= threshold]
        c.check(g, f"threshold {threshold}: recomputed from predictions",
                (len(sub), round(sub["correct"].mean(), 3)), (covered, round(accuracy, 3)))


def check_noise(c: Checker) -> None:
    g = "Robustness under corruption (Figure 8)"
    df = load("single_stream", "noise_results.csv")
    acc = df.groupby(["noise_kind", "noise_level"])["correct"].mean()
    c.check(g, "clean accuracy", acc[("gaussian", 0.0)], 0.970)
    for sigma, expected in ((5.0, 0.866), (10.0, 0.748), (25.0, 0.614), (50.0, 0.510)):
        c.check(g, f"Gaussian sigma={sigma:g}", acc[("gaussian", sigma)], expected)
    for amount, expected in ((0.01, 0.916), (0.03, 0.876), (0.07, 0.738)):
        c.check(g, f"salt-and-pepper {amount:.0%}", acc[("saltpepper", amount)], expected)
    c.check(g, "images per corruption level",
            int(df.groupby(["noise_kind", "noise_level"]).size().max()), 202)


def check_rarity(c: Checker) -> None:
    g = "Rarity analysis (Section 3.4)"
    df = load("single_stream", "rarity_f1_vs_ntrain.csv")
    c.check(g, "species scored", len(df), 31)
    imperfect = df[df["f1"] < 1.0]
    c.check(g, "species not classified perfectly", len(imperfect), 4)
    c.check(g, "every species under 180 training images is error-free",
            bool((df[df["n_train"] < 180]["f1"] == 1.0).all()), True)
    c.check(g, "lowest training count among imperfect species", int(imperfect["n_train"].min()), 180)
    c.check(g, "highest training count among imperfect species", int(imperfect["n_train"].max()), 345)
    # Bigenerina nodosaria is error-free on a single test image, which is too weak to lean on.
    bn = df[df["species"] == "Bigenerina nodosaria"].iloc[0]
    c.check(g, "Bigenerina nodosaria test support", int(bn["support"]), 1)


def check_training(c: Checker) -> None:
    g = "Training dynamics (Figure 4, Section 3.2)"
    single = load("single_stream", "training_metrics.csv")
    c.check(g, "single-stream epochs before early stopping", len(single), 7)
    c.check(g, "single-stream training loss at epoch 1", single["train_loss"].iloc[0], 1.50, places=2)
    c.check(g, "single-stream training loss at epoch 7", single["train_loss"].iloc[-1], 0.99, places=2)
    c.check(g, "single-stream training accuracy at epoch 1", single["train_acc"].iloc[0], 0.760)
    c.check(g, "single-stream training accuracy at epoch 7", single["train_acc"].iloc[-1], 0.935)
    c.check(g, "single-stream peak validation accuracy", single["val_acc"].max(), 0.965)
    c.check(g, "single-stream best epoch", int(single["val_acc"].idxmax()) + 1, 2)

    dual = load("dual_stream", "training_metrics.csv")
    c.check(g, "dual-stream peak validation accuracy", dual["val_acc"].max(), 0.965)
    c.check(g, "dual-stream best epoch", int(dual["val_acc"].idxmax()) + 1, 3)
    # The teacher fits faster than the student from the first epoch, which is the reported
    # mechanism behind the dual-stream arm underperforming.
    c.check(g, "teacher loss at epoch 1", dual["train_teacher_loss"].iloc[0], 0.93, places=2)
    c.check(g, "student loss at epoch 1", dual["train_student_loss"].iloc[0], 1.45, places=2)


def check_firewall(c: Checker) -> None:
    g = "Evaluation firewall (Section 3.3)"
    for arm in ("single_stream", "dual_stream"):
        name = arm.replace("_", "-")
        prov = load(arm, "selection_provenance.json")
        c.check(g, f"{name} selection metric", prov["selection_metric"], "val_acc")
        c.check(g, f"{name} selection split", prov["selection_split"], "val")
        c.check(g, f"{name} audit cohort used for selection", prov["audit_cohort_used"], False)
        c.check(g, f"{name} test split used for selection", prov["test_split_used"], False)

        # The selected epoch must be the argmax of validation accuracy over the candidates.
        table = load(arm, "checkpoint_selection_val.csv")
        best = int(table.loc[table["val_acc"].idxmax(), "epoch"])
        c.check(g, f"{name} selected epoch is the validation argmax", prov["selected_epoch"], best)

    c.check(g, "single-stream deployed epoch", load("single_stream", "selection_provenance.json")["selected_epoch"], 2)
    c.check(g, "dual-stream deployed epoch", load("dual_stream", "selection_provenance.json")["selected_epoch"], 3)


def check_helegans(c: Checker) -> None:
    g = "Hoeglundina elegans leave-one-specimen-out (Section 3.6)"
    probe = load("single_stream", "hoeglundina_elegans_probe.json")
    c.check(g, "specimens of H. elegans in the dataset", probe["n_specimens_in_dataset"], 2)
    c.check(g, "held-out-view F1 for H. elegans", probe["held_out_view_metrics"]["f1-score"], 1.000)
    folds = {f["fold"]: f for f in probe["loso"]["folds"]}
    c.check(g, "accuracy holding out specimen 0", folds["hold_out_0"]["accuracy"], 1.000)
    c.check(g, "accuracy holding out specimen 28", folds["hold_out_28"]["accuracy"], 0.738)


GROUPS = {
    "dataset": check_dataset,
    "test": check_test_metrics,
    "calibration": check_calibration,
    "audit": check_audit,
    "coverage": check_coverage,
    "noise": check_noise,
    "rarity": check_rarity,
    "training": check_training,
    "firewall": check_firewall,
    "helegans": check_helegans,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=sorted(GROUPS), help="run one group of checks")
    args = ap.parse_args()

    c = Checker()
    for name, fn in GROUPS.items():
        if args.group in (None, name):
            fn(c)

    print("Verifying the reported numbers of AIIG-D-26-00130 against the released outputs.")
    sys.exit(c.report(None))


if __name__ == "__main__":
    main()
