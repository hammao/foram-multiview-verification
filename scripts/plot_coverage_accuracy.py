"""Coverage-accuracy curve on the locked audit cohort, for the deployed checkpoint only.

Reads the summary written by `audit_confidence_curve.py`. Adds no selection step: the curve is
a description of one frozen checkpoint, not a way to pick one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

# Thresholds called out in the Figure 6 caption.
ANNOTATED = (0.2, 0.5, 0.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    curve = sorted(summary["coverage_accuracy_curve"], key=lambda r: r["coverage"])
    n_total = int(summary["n"])

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=300)
    ax.plot(
        [r["coverage"] for r in curve],
        [r["accuracy_on_covered"] for r in curve],
        "o-",
        color="#1f4e79",
        lw=1.8,
        ms=6,
    )

    for r in curve:
        if r["threshold"] not in ANNOTATED:
            continue
        # Keep labels clear of the curve: below-right at full coverage, above-left elsewhere.
        offset = (-10, -22) if r["coverage"] > 0.95 else (8, 8)
        ax.annotate(
            f"t = {r['threshold']:.1f}\n{r['n_covered']}/{n_total} covered\nacc = {r['accuracy_on_covered']:.3f}",
            (r["coverage"], r["accuracy_on_covered"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=7.5,
            color="#1f4e79",
            ha="right" if r["coverage"] > 0.95 else "left",
        )

    ax.set_xlabel(f"Coverage (fraction of the {n_total}-image locked audit cohort receiving a prediction)")
    ax.set_ylabel("Top-1 accuracy on covered images")
    ax.set_xlim(0.22, 1.06)
    ax.set_ylim(0.34, 0.82)
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.99,
        0.02,
        f"deployed single-stream checkpoint; temperature-scaled confidence (T = {summary['temperature']:.3f})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
