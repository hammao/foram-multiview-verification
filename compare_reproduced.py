#!/usr/bin/env python3
"""Diff freshly reproduced evaluation outputs against the committed ones.

Called by ``reproduce_eval.sh`` after re-running evaluation from the released weights. Metrics
are compared to four decimal places rather than exactly, because cuDNN kernel selection and
CPU-versus-GPU reductions perturb the last bits of a float without changing any prediction.
Predictions themselves are compared exactly: a single differing label is a real difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TOL = 1e-4


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= TOL
    return a == b


def compare_metrics(label: str, fresh: dict, committed: dict, keys) -> list[str]:
    problems = []
    for key in keys:
        if key not in fresh or key not in committed:
            continue
        if not close(fresh[key], committed[key]):
            problems.append(f"{label}: {key} reproduced {fresh[key]!r}, committed {committed[key]!r}")
    return problems


def compare_predictions(label: str, fresh: Path, committed: Path, key: str, cols) -> list[str]:
    a = pd.read_csv(fresh).sort_values(key).reset_index(drop=True)
    b = pd.read_csv(committed).sort_values(key).reset_index(drop=True)
    if len(a) != len(b):
        return [f"{label}: {len(a)} rows reproduced, {len(b)} committed"]
    problems = []
    for col in cols:
        if col not in a.columns or col not in b.columns:
            continue
        mismatched = int((a[col].astype(str) != b[col].astype(str)).sum())
        if mismatched:
            problems.append(f"{label}: {mismatched} of {len(a)} rows differ in {col!r}")
    return problems


def main() -> None:
    arm = sys.argv[1]
    repro = ROOT / "reproduced" / arm
    committed = ROOT / "results" / arm
    problems: list[str] = []

    report_name = f"eval_report_{arm}.json"
    fresh_report = repro / "manifest" / report_name
    if fresh_report.exists():
        problems += compare_metrics(
            "test metrics",
            load_json(fresh_report),
            load_json(committed / "eval_report.json"),
            ("temperature", "ece", "ece_calibrated", "nll", "nll_calibrated"),
        )
        fresh = load_json(fresh_report)["report"]
        old = load_json(committed / "eval_report.json")["report"]
        problems += compare_metrics("test accuracy", fresh, old, ("accuracy",))
        for avg in ("macro avg", "weighted avg"):
            problems += compare_metrics(f"test {avg}", fresh[avg], old[avg], ("precision", "recall", "f1-score"))
        problems += compare_predictions(
            "test predictions",
            repro / "manifest" / report_name.replace(".json", "_predictions.csv"),
            committed / "test_predictions.csv",
            "path_resized",
            ("species", "predicted", "correct"),
        )
    else:
        problems.append(f"missing reproduced report: {fresh_report}")

    fresh_temp = repro / "temperature.json"
    if fresh_temp.exists():
        problems += compare_metrics(
            "calibration",
            load_json(fresh_temp),
            load_json(committed / "temperature.json"),
            ("temperature", "ece_before", "ece_after", "nll_before", "nll_after"),
        )

    fresh_audit = repro / "audit_once_summary.json"
    if fresh_audit.exists():
        a, b = load_json(fresh_audit), load_json(committed / "audit_summary.json")
        problems += compare_metrics("audit", a, b, ("n",))
        for stratum in ("pooled", "normal", "difficult"):
            if stratum in a and stratum in b:
                problems += compare_metrics(
                    f"audit {stratum}", a[stratum], b[stratum],
                    ("top1", "top3", "top1_wilson_lo", "top1_wilson_hi"),
                )
        problems += compare_predictions(
            "audit predictions",
            repro / "audit_once_predictions.csv",
            committed / "audit_predictions.csv",
            "image_path",
            ("true_species", "pred_1", "pred_2", "pred_3"),
        )
    else:
        problems.append(f"missing reproduced audit summary: {fresh_audit}")

    if problems:
        print(f"{arm}: REPRODUCTION DIFFERS")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print(f"{arm}: reproduced outputs match the committed ones (metrics to {TOL:g}, predictions exactly)")


if __name__ == "__main__":
    main()
