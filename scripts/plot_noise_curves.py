"""Replot noise robustness without an embedded title or a clipped legend."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARMS = [
    ("single_stream", "Single-stream multi-view", "o-", "#1f4e79"),
    ("dual_stream", "Dual-stream multi-view", "s--", "#a6611a"),
]
KINDS = [
    ("gaussian", "Gaussian noise standard deviation (8-bit levels)", "(a)"),
    ("saltpepper", "Salt-and-pepper corruption fraction", "(b)"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, nargs="+", help="one results.csv per arm, in ARMS order")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = {arm[0]: pd.read_csv(path) for arm, path in zip(ARMS, args.results, strict=True)}

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    n_images = 0
    for ax, (kind, xlabel, panel) in zip(axes, KINDS, strict=True):
        for key, label, style, colour in ARMS:
            sub = frames[key][frames[key]["noise_kind"] == kind]
            if sub.empty:
                continue
            acc = sub.groupby("noise_level")["correct"].mean()
            n_images = max(n_images, int(sub.groupby("noise_level").size().max()))
            ax.plot(acc.index, acc.values, style, color=colour, label=label, lw=1.6, ms=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 accuracy")
        ax.set_ylim(0.0, 1.02)
        ax.tick_params(labelleft=True)
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_title(panel, loc="left", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(fontsize=8, loc="lower left", frameon=False)
    axes[1].text(0.98, 0.04, f"n = {n_images} held-out-view test images", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=7.5, color="0.35")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
