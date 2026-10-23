#!/usr/bin/env python3
"""Figures for the 50-species paper, built from the frozen split and the battery outputs.

  fig2_dataset_composition.png   images per species by source and tier, grouped by wall type
  fig4_training_curves.png       SS and DS loss / accuracy per epoch, selected epoch marked
  fig5_confusion_test.png        row-normalised confusion on the specimen test split (SS)
  fig6_coverage_accuracy.png     confidence gate on the frozen test split and the Leica overlap
  fig8_noise_curves.png          Gaussian and salt-and-pepper degradation on the full test split
  fig_rarity_f1.png              per-species F1 against training shells and images
  fig_ss_vs_ds.png               per-shell outcome of the two arms on the test split

Figures whose inputs are not yet present are skipped, so the script can be rerun as results land.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports" / "experiments" / "2026-10_drive140_paper"
FIG = REP / "figures"
SPLIT = ROOT / "processed" / "drive140" / "manifest" / "splits_specimen_tvt"
METRICS = ROOT / "processed" / "drive140" / "manifest"
MODELS = ROOT / "models" / "v3_expansion" / "20261016_drive140_paper"
WALL_ORDER = ["hyaline", "agglutinated", "porcelaneous"]
WALL_COLOR = {"hyaline": "#4C72B0", "agglutinated": "#DD8452", "porcelaneous": "#55A868"}
SRC_LABEL = {"edof": "EDOF composite", "gif_original": "GIF frame"}
SRC_GROUP = {"stacked_tiff": "edof", "pmax_tiff": "edof", "gif_original": "gif_original"}
VIEWS = ROOT / "processed" / "drive140" / "manifest" / "views_resized.csv"
COMPLETENESS = ROOT / "reports" / "ops" / "drive_species_completeness_20261015.csv"
FIG1_SPECIES = ["Amphistegina radiata", "Miliolinella circularis", "Operculina ammonoides", "Pyrgo depressa", "Quinqueloculina pulchella",
                "Sahulia kerimbaensis", "Sorites orbiculus", "Spiroloculina communis", "Spirosigmoilina parri", "Triloculina trigonula"]
DPI = 300

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def panel(ax, letter: str, x: float = -0.02, y: float = 1.01) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def table1() -> pd.DataFrame:
    return pd.read_csv(REP / "table1_species_taxonomy.csv")


def italic(name: str) -> str:
    """Genus + epithet in italics; OTU codes and 'sp.' upright."""
    parts = name.split()
    if len(parts) >= 2 and parts[1] != "sp.":
        return rf"$\it{{{parts[0]}}}$ $\it{{{parts[1]}}}$" + (" " + " ".join(parts[2:]) if len(parts) > 2 else "")
    return rf"$\it{{{parts[0]}}}$ " + " ".join(parts[1:])


def fig_s3_per_species() -> None:
    t = table1()
    frames = {s: pd.read_csv(SPLIT / f"{s}.csv") for s in ("train", "val", "test")}
    allv = pd.concat([f.assign(tier=s) for s, f in frames.items()])
    t["_w"] = t["wall_type"].map(WALL_ORDER.index)
    t = t.sort_values(["_w", "n_images_total", "display_name"], ascending=[True, False, True]).reset_index(drop=True)
    order = t["code_label"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.6), gridspec_kw={"width_ratios": [3, 2]}, sharey=True)
    y = np.arange(len(order))
    # left: by source
    left = np.zeros(len(order))
    allv["src"] = allv.source.map(SRC_GROUP)
    for src, lab in SRC_LABEL.items():
        counts = allv[allv.src == src].groupby("species").size().reindex(order).fillna(0).to_numpy()
        axes[0].barh(y, counts, left=left, label=lab, height=0.75, color={"edof": "#4C72B0", "gif_original": "#55A868"}[src])
        left += counts
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([italic(n) for n in t["display_name"]], fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Images per species, by source")
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    for yi, w in zip(y, t["wall_type"], strict=True):
        axes[0].get_yticklabels()[yi].set_color(WALL_COLOR[w])
    # right: by tier
    left = np.zeros(len(order))
    for tier, col in (("train", "#999999"), ("val", "#f0ad4e"), ("test", "#c0392b")):
        counts = allv[allv.tier == tier].groupby("species").size().reindex(order).fillna(0).to_numpy()
        axes[1].barh(y, counts, left=left, label=f"{tier} ({frames[tier].groupby(['species','specimen_id']).ngroups} shells)", color=col, height=0.75)
        left += counts
    for yi, n in zip(y, t["n_specimens"], strict=True):
        axes[1].text(left[yi] + 1, yi, f"{n}", va="center", fontsize=5.5, color="#333")
    axes[1].set_xlabel("Images per species, by split tier\n(number = complete shells)")
    axes[1].legend(frameon=False, loc="lower right", fontsize=7)
    handles = [plt.Line2D([], [], color=WALL_COLOR[w], lw=4) for w in WALL_ORDER]
    fig.legend(handles, [w.capitalize() for w in WALL_ORDER], loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.995))
    panel(axes[0], "A", x=0.0, y=1.005)
    panel(axes[1], "B", x=0.0, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "fig_s3_per_species_composition.png", dpi=DPI)
    plt.close(fig)
    print("fig_s3 written")


def _tile(path: Path, size: int = 300) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def _reference_specimen(views: pd.DataFrame, species: str) -> int:
    train = pd.read_csv(SPLIT / "train.csv")
    ids = sorted(train[train.species == species].specimen_id.unique())
    return int(ids[0]) if ids else int(sorted(views[views.species == species].specimen_id.unique())[0])


def _view_path(views: pd.DataFrame, species: str, sid: int, view_id: int) -> Path | None:
    part = views[(views.species == species) & (views.specimen_id == sid) & (views.view_id == view_id)]
    for src in ("pmax_tiff", "stacked_tiff", "gif_original"):
        hit = part[part.source == src]
        if not hit.empty:
            return Path(str(hit.iloc[0].path_resized))
    return None


def fig1() -> None:
    """Six rotational views (EDOF composites) of the reference specimen of ten species."""
    views = pd.read_csv(VIEWS)
    t = table1().set_index("code_label")
    rows = [sp for sp in FIG1_SPECIES if sp in t.index]
    fig, axes = plt.subplots(len(rows), 6, figsize=(6.6, 1.05 * len(rows) + 0.3))
    for i, sp in enumerate(rows):
        sid = _reference_specimen(views, sp)
        for j in range(6):
            ax = axes[i, j]
            p = _view_path(views, sp, sid, j + 1)
            if p is not None and p.exists():
                ax.imshow(_tile(p))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(True)
                sp_.set_linewidth(0.4)
                sp_.set_color("#999")
            if i == 0:
                ax.set_title(f"{60 * j}°", fontsize=8, pad=3)
        axes[i, 0].set_ylabel(italic(t.loc[sp, "display_name"]), fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(left=0.24, right=0.995, top=0.96, bottom=0.01, wspace=0.04, hspace=0.06)
    fig.savefig(FIG / "fig1_rotational_views.png", dpi=DPI)
    plt.close(fig)
    print("fig1 written")


def fig2() -> None:
    """Dataset dashboard: totals, composition by source, images by tier, specimens per species."""
    t = table1()
    frames = {s: pd.read_csv(SPLIT / f"{s}.csv") for s in ("train", "val", "test")}
    allv = pd.concat([f.assign(tier=s) for s, f in frames.items()])
    allv["src"] = allv.source.map(SRC_GROUP)
    n_img, n_spp = len(allv), allv.species.nunique()
    n_shells = allv.groupby(["species", "specimen_id"]).ngroups
    n_angles = allv.groupby(["species", "specimen_id", "view_id"]).ngroups
    by_src = allv.groupby("src").size()
    # focal-plane slices acquired behind the EDOF composites (inventory of the Drive specimen folders)
    n_slices = None
    if COMPLETENESS.exists():
        comp = pd.read_csv(COMPLETENESS)
        comp = comp[comp.complete.astype(str).str.lower() == "true"]
        n_slices = int(comp.z_files.sum())
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]}):
        fig = plt.figure(figsize=(7.2, 4.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.25, 1], hspace=0.75, wspace=0.35, left=0.06, right=0.98, top=0.93, bottom=0.12)
        # A: headline numbers
        ax = fig.add_subplot(gs[0, 0])
        ax.axis("off")
        ax.text(0, 0.98, "Reference catalogue (complete specimens only)", fontsize=9, fontweight="bold", va="top")
        ax.text(0, 0.78, f"{n_img:,}", fontsize=30, fontweight="bold", va="top")
        ax.text(0, 0.30, "images", fontsize=9, va="top")
        ax.text(0, 0.12, f"{n_spp} species  •  {n_shells} specimens  •  {n_angles} specimen-angles (6-view rotations)", fontsize=7, va="top")
        ax.text(0, -0.06, f"Every specimen contributes one EDOF composite and one GIF frame per view; {by_src['edof']:,} EDOF composites, {by_src['gif_original']:,} GIF frames.", fontsize=6.5, va="top", color="#1f4e79")
        if n_slices:
            ax.text(0, -0.24, f"Acquired behind the composites: {n_slices:,} focal-plane Z-slices ({n_slices / n_angles:.0f} per view on average),\nused for compositing only, not as training images.", fontsize=6.5, va="top", color="#1f4e79", linespacing=1.4)
        # B: donut by source
        ax = fig.add_subplot(gs[0, 1])
        vals = [by_src["edof"], by_src["gif_original"]]
        cols = ["#4C72B0", "#55A868"]
        ax.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.38, edgecolor="white"))
        ax.set_title("Composition by source", fontsize=8, fontweight="bold")
        ax.legend([f"EDOF composites: {vals[0]:,}", f"GIF frames: {vals[1]:,}"], loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=True, fontsize=6.5, edgecolor="#ccc", ncol=1)
        # C: images by partition tier (with shells)
        ax = fig.add_subplot(gs[1, 0])
        tiers = ["train", "val", "test"]
        cnt = [len(frames[k]) for k in tiers]
        sh = [frames[k].groupby(["species", "specimen_id"]).ngroups for k in tiers]
        spp = [frames[k].species.nunique() for k in tiers]
        bars = ax.bar(["Train", "Validation", "Test"], cnt, color=["#999999", "#f0ad4e", "#c0392b"], width=0.6)
        for b, c, s_, sp_ in zip(bars, cnt, sh, spp, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, c + 15, f"{c:,} images\n{s_} shells, {sp_} spp.", ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, max(cnt) * 1.3)
        ax.set_ylabel("Images", fontsize=7)
        ax.set_title("Partition by specimen (frozen before training)", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        # D: specimens per species
        ax = fig.add_subplot(gs[1, 1])
        mult = t["n_specimens"].value_counts().sort_index()
        ax.bar(mult.index, mult.values, color="#4C72B0", width=0.7)
        for k, v in mult.items():
            ax.text(k, v + 0.4, str(v), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(list(mult.index))
        ax.set_xlabel("Complete specimens per species", fontsize=7)
        ax.set_ylabel("Species", fontsize=7)
        ax.set_ylim(0, mult.max() * 1.25)
        ax.set_title("Specimens per species", fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        for ax_, letter in zip(fig.axes, "ABCD", strict=True):
            ax_.text(-0.08 if letter != "A" else 0.0, 1.08, letter, transform=ax_.transAxes, fontsize=10, fontweight="bold", ha="right" if letter != "A" else "left", va="bottom", family="sans-serif")
        fig.savefig(FIG / "fig2_dataset_composition.png", dpi=DPI)
        plt.close(fig)
    print("fig2 written")


def fig7(arm: str = "single_stream") -> None:
    """PCA of penultimate-layer embeddings of the test tier, species centroids numbered and keyed."""
    p = REP / arm / "embedding_pca_test.csv"
    if not p.exists():
        print("fig7 skipped (run scripts/extract_embeddings_drive140.py)")
        return
    d = pd.read_csv(p)
    meta = json.loads((REP / arm / "embedding_pca_test_meta.json").read_text())
    t = table1().set_index("code_label")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cent = d.groupby("species")[["PC1", "PC2"]].mean()
    for w in WALL_ORDER:
        m = d.wall_type == w
        ax.scatter(d.loc[m, "PC1"], d.loc[m, "PC2"], s=14, color=WALL_COLOR[w], alpha=0.55, edgecolor="none", label=f"{w.capitalize()} ({m.sum()} images)")
    # number every species centroid
    order = cent.assign(w=cent.index.map(lambda s_: WALL_ORDER.index(t.loc[s_, "wall_type"]))).sort_values(["w", "PC1"])
    numbers = {sp: i + 1 for i, sp in enumerate(order.index)}
    # nudge near-coincident centroid labels apart so that both numbers stay legible
    lab = cent.copy()
    sx, sy = d.PC1.max() - d.PC1.min(), d.PC2.max() - d.PC2.min()
    for _ in range(30):
        moved = False
        for a_ in lab.index:
            for b_ in lab.index:
                if a_ >= b_:
                    continue
                dx, dy = (lab.loc[a_, "PC1"] - lab.loc[b_, "PC1"]) / sx, (lab.loc[a_, "PC2"] - lab.loc[b_, "PC2"]) / sy
                dist = np.hypot(dx, dy)
                if dist < 0.045:
                    push = (0.045 - dist) / 2 + 1e-3
                    ux, uy = (dx / dist, dy / dist) if dist > 1e-9 else (0.0, 1.0)
                    lab.loc[a_, "PC1"] += ux * push * sx
                    lab.loc[a_, "PC2"] += uy * push * sy
                    lab.loc[b_, "PC1"] -= ux * push * sx
                    lab.loc[b_, "PC2"] -= uy * push * sy
                    moved = True
        if not moved:
            break
    for sp, r in lab.iterrows():
        ax.text(r.PC1, r.PC2, str(numbers[sp]), fontsize=5.5, ha="center", va="center", fontweight="bold", color="black",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=WALL_COLOR[t.loc[sp, "wall_type"]], lw=0.6, alpha=0.95))
    ax.margins(0.06)
    ax.set_xlabel(f"PC1 ({100 * meta['explained_variance_pc1']:.1f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * meta['explained_variance_pc2']:.1f}% of variance)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3)
    # key to numbers, outside the axes on the right: one line per species, number upright, binomial in italics
    n_key = len(order.index)
    y_top, y_bot = 0.88, 0.14
    for i, sp in enumerate(order.index):
        y = y_top - i * (y_top - y_bot) / (n_key - 1)
        fig.text(0.745, y, f"{numbers[sp]:>2}", fontsize=6, va="center", ha="right", family="DejaVu Sans Mono", color=WALL_COLOR[t.loc[sp, "wall_type"]], fontweight="bold")
        fig.text(0.755, y, italic(t.loc[sp, "display_name"]), fontsize=6, va="center", ha="left")
    fig.subplots_adjust(left=0.08, right=0.72, top=0.955, bottom=0.08)
    fig.savefig(FIG / "fig7_embedding_pca.png", dpi=DPI)
    plt.close(fig)
    pd.Series(numbers).rename("number").to_csv(REP / arm / "embedding_pca_species_numbers.csv")
    print("fig7 written")


def fig3() -> None:
    """Workflow diagram, same three bands as the Oct2026 figure, with the specimen-tier design."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    bands = [
        (38, 56, "#eef3f9", "DATA ENGINEERING"),
        (20, 36, "#f6f0f6", "MODEL DEVELOPMENT"),
        (2, 18, "#eef7f1", "DEPLOYMENT AND VALIDATION"),
    ]
    for i, (y0, y1, col, label) in enumerate(bands):
        ax.add_patch(FancyBboxPatch((1, y0), 98, y1 - y0, boxstyle="round,pad=0.4", fc=col, ec="#cfd6dd", lw=0.8))
        if i == 1:
            ax.text(3, y1 - 2.2, label, fontsize=7.5, weight="bold", va="top", ha="left")
        else:
            ax.text(97, y1 - 2.2, label, fontsize=7.5, weight="bold", va="top", ha="right")

    def box(x, y, w, h, title, sub):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc="white", ec="#9aa5b1", lw=0.8))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=6.8, weight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.2)
        return (x, y, w, h)

    def arrow(a, b, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=0.9, color="#333", linestyle=ls, shrinkA=2, shrinkB=2))

    r1 = [
        box(3, 40, 17, 10, "Acquisition", "6 views × (EDOF + GIF)\n140 specimens, 50 spp"),
        box(22, 40, 17, 10, "Completeness gate", "all 6 views in\nboth modalities"),
        box(41, 40, 17, 10, "Manifest", "specimen · view · source\nWoRMS-harmonised label"),
        box(60, 40, 17, 10, "Preprocess", "Otsu crop · letterbox\n384 × 384 PNG"),
        box(79, 40, 18, 10, "Specimen tiers", "multiplicity rule\ntrain / val / test, frozen"),
    ]
    for a, b in zip(r1[:-1], r1[1:], strict=True):
        arrow((a[0] + a[2], a[1] + 5), (b[0], b[1] + 5))
    r2 = [
        box(60, 22, 37, 10, "Training (40 epochs)", "ConvNeXt-Base + genus-aware soft targets\nSS-MV, or DS-MV with specimen-level teacher"),
        box(32, 22, 25, 10, "Selection + calibration", "earliest best validation epoch\ntemperature fitted on validation"),
        box(3, 22, 26, 10, "Scored once", "frozen test tier (33 new shells)\nexternal Leica cohort (39 images)"),
    ]
    arrow((88, 40), (88, 32))
    arrow((60, 27), (57, 27))
    arrow((32, 27), (29, 27))
    r3 = [
        box(3, 4, 26, 10, "Single-image inference", "same preprocessor\ncalibrated softmax"),
        box(32, 4, 30, 10, "Decision policy", "accept ≥ 0.9 · suggest · unknown < 0.2\nnearest-genus fallback (Genus sp.)"),
        box(65, 4, 32, 10, "Nomenclature", "WoRMS AphiaID lookup\nopen-nomenclature codes retained"),
    ]
    arrow((44, 22), (16, 14), ls="--")
    arrow((29, 9), (32, 9))
    arrow((62, 9), (65, 9))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pipeline.png", dpi=DPI)
    plt.close(fig)
    print("fig3 written")


def fig4() -> None:
    arms = {
        "SS-MV": (METRICS / "training_metrics_20261016_paper_single_stream.csv", MODELS / "single_stream" / "run_status.json"),
        "DS-MV": (METRICS / "training_metrics_20261016_paper_dual_stream.csv", MODELS / "dual_stream" / "run_status.json"),
    }
    present = {k: v for k, v in arms.items() if v[0].exists()}
    if not present:
        print("fig4 skipped (no metrics yet)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for (name, (csv, status)), ls in zip(present.items(), ("-", "--"), strict=False):
        m = pd.read_csv(csv)
        sel = json.loads(status.read_text()).get("selected_epoch") if status.exists() else None
        # student (classification) loss is the comparable quantity; the DS total also contains teacher and distillation terms
        student = m["train_student_loss"] if "train_student_loss" in m else m["train_loss"]
        axes[0].plot(m.epoch, student, ls, color="#444", label=f"{name} student loss")
        if "train_teacher_loss" in m:
            axes[0].plot(m.epoch, m.train_teacher_loss, ls, color="#4C72B0", label=f"{name} teacher loss")
            axes[0].plot(m.epoch, m.train_distill_loss, ls, color="#c0392b", label=f"{name} distillation loss")
        axes[1].plot(m.epoch, m.train_acc, ls, color="#999", label=f"{name} train")
        axes[1].plot(m.epoch, m.val_acc, ls, color="#c0392b", label=f"{name} validation")
        axes[1].plot(m.epoch, m.val_genus_acc, ls, color="#4C72B0", label=f"{name} validation (genus)")
        if sel:
            axes[1].axvline(sel, color="#c0392b", lw=0.6, alpha=0.5)
            axes[1].annotate(f"{name}\nepoch {sel}", (sel, 1.0), xytext=(-4 if name == "DS-MV" else 4, -2), textcoords="offset points", ha="right" if name == "DS-MV" else "left", va="top", fontsize=5.5, color="#c0392b")
    axes[0].axvspan(0.5, 5.5, color="#eee", zorder=0)
    axes[1].axvspan(0.5, 5.5, color="#eee", zorder=0)
    axes[0].text(3, axes[0].get_ylim()[1] * 0.95, "backbone\nfrozen", ha="center", va="top", fontsize=6, color="#666")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[0].set_ylim(0, 3.5)
    axes[0].legend(frameon=False, fontsize=5.5)
    axes[1].legend(frameon=False, fontsize=5.5, loc="lower right", ncol=2)
    panel(axes[0], "A", x=-0.14)
    panel(axes[1], "B", x=-0.14)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_training_curves.png", dpi=DPI)
    plt.close(fig)
    print("fig4 written")


def fig5(arm: str = "single_stream") -> None:
    p = REP / arm / "test_predictions.csv"
    if not p.exists():
        print("fig5 skipped")
        return
    pred = pd.read_csv(p)
    t = table1().set_index("code_label")
    rows = sorted(pred.species.unique(), key=lambda s: (WALL_ORDER.index(t.loc[s, "wall_type"]), t.loc[s, "family"], s))
    cols = rows + sorted(set(pred.pred_1) - set(rows), key=lambda s: (WALL_ORDER.index(t.loc[s, "wall_type"]), s))
    cm = pd.crosstab(pred.species, pred.pred_1).reindex(index=rows, columns=cols).fillna(0)
    norm = cm.div(cm.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = int(cm.iat[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=5, color="white" if norm.iat[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([italic(t.loc[c, "display_name"]) for c in cols], rotation=90, fontsize=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([italic(t.loc[r, "display_name"]) for r in rows], fontsize=5)
    for lab, name in zip(ax.get_yticklabels(), rows, strict=True):
        lab.set_color(WALL_COLOR[t.loc[name, "wall_type"]])
    for lab, name in zip(ax.get_xticklabels(), cols, strict=True):
        lab.set_color(WALL_COLOR[t.loc[name, "wall_type"]])
    # wall-type separators
    for seq, axis in ((rows, "y"), (cols, "x")):
        walls = [t.loc[s, "wall_type"] for s in seq]
        for k in range(1, len(seq)):
            if walls[k] != walls[k - 1]:
                (ax.axhline if axis == "y" else ax.axvline)(k - 0.5, color="#888", lw=0.6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (held-out shell)")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_confusion_test.png", dpi=DPI)
    plt.close(fig)
    print("fig5 written")


def fig6(arm: str = "single_stream") -> None:
    test_curve = REP / arm / "coverage_accuracy_test.csv"
    audit_pred = REP / arm / "audit_confidence_39_predictions.csv"
    if not test_curve.exists():
        print("fig6 skipped")
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    tc = pd.read_csv(test_curve)
    ax.plot(tc.coverage, tc.accuracy_on_covered, "o-", color="#c0392b", label=f"Specimen test (n = {int(tc.n_covered.max())})")
    for _, r in tc.iterrows():
        if r.threshold in (0.5, 0.7, 0.9):
            ax.annotate(f"≥{r.threshold:.1f}", (r.coverage, r.accuracy_on_covered), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=6, color="#c0392b", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    if audit_pred.exists():
        a = pd.read_csv(audit_pred)
        pts = []
        for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
            cov = a[a.confidence >= th]
            if len(cov):
                pts.append((len(cov) / len(a), cov.correct.mean(), th))
        xs, ys, ths = zip(*pts, strict=True)
        ax.plot(xs, ys, "s--", color="#4C72B0", label=f"Leica overlap (n = {len(a)})")
        for x, y_, th in pts:
            if th in (0.5, 0.7, 0.9):
                ax.annotate(f"≥{th:.1f}", (x, y_), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=6, color="#4C72B0", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.set_xlabel("Coverage (fraction of images returned)")
    ax.set_ylabel("Top-1 accuracy on covered images")
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.06)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_coverage_accuracy.png", dpi=DPI)
    plt.close(fig)
    print("fig6 written")


def fig8() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    series: dict[str, tuple[pd.Series, pd.Series]] = {}
    for arm, name in (("single_stream", "SS-MV"), ("dual_stream", "DS-MV")):
        runs = sorted((REP / arm / "noise").glob("noise_test_*/results.csv")) if (REP / arm / "noise").exists() else []
        if runs:
            r = pd.read_csv(runs[-1])
            series[name] = (r[r.noise_kind == "gaussian"].groupby("noise_level").correct.mean(), r[r.noise_kind == "saltpepper"].groupby("noise_level").correct.mean())
    if not series:
        plt.close(fig)
        print("fig8 skipped")
        return
    style = {"SS-MV": ("o-", "#1f77b4"), "DS-MV": ("s--", "#ff7f0e")}
    for name, (g, sp) in series.items():
        ls, col = style[name]
        axes[0].plot(g.index, g.values, ls, color=col, label=name)
        axes[1].plot(sp.index * 100, sp.values, ls, color=col, label=name)
    other = {"SS-MV": "DS-MV", "DS-MV": "SS-MV"}
    for k, ax, scale in ((0, axes[0], 1), (1, axes[1], 100)):
        for name, ser_pair in series.items():
            ser = ser_pair[k]
            comp = series[other[name]][k] if other[name] in series else None
            for x, v in ser.items():
                above = comp is None or v >= comp.get(x, -1)
                ax.annotate(f"{v:.3f}", (x * scale, v), textcoords="offset points", xytext=(0, 6 if above else -7), ha="center", va="bottom" if above else "top", fontsize=5.5, color=style[name][1])
    axes[0].set_xlabel("Gaussian noise σ (8-bit units)")
    axes[1].set_xlabel("Salt-and-pepper corruption (% pixels)")
    for ax in axes:
        ax.set_ylabel("Top-1 accuracy, full test split")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7)
    panel(axes[0], "A", x=-0.14)
    panel(axes[1], "B", x=-0.14)
    fig.tight_layout()
    fig.savefig(FIG / "fig8_noise_curves.png", dpi=DPI)
    plt.close(fig)
    print("fig8 written")


def rarity_table(arm: str = "single_stream") -> pd.DataFrame | None:
    """Every held-out shell (validation + test) with the number of training individuals of its species."""
    files = {s: REP / arm / f"{s}_specimens.csv" for s in ("val", "test")}
    if not all(p.exists() for p in files.values()):
        return None
    ntr = pd.read_csv(SPLIT / "train.csv").groupby("species").specimen_id.nunique()
    nimg = pd.read_csv(SPLIT / "train.csv").groupby("species").size()
    t = table1().set_index("code_label")
    d = pd.concat([pd.read_csv(p).assign(tier=s) for s, p in files.items()], ignore_index=True)
    d["n_train_shells"] = d.species.map(ntr)
    d["n_train_images"] = d.species.map(nimg)
    d["wall_type"] = d.species.map(t.wall_type)
    d["display_name"] = d.species.map(t.display_name)
    return d


def fig_rarity(arm: str = "single_stream") -> None:
    d = rarity_table(arm)
    if d is None:
        print("rarity skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    rng = np.random.default_rng(1337)
    jit = rng.uniform(-0.15, 0.15, len(d))
    ax = axes[0]
    for w in WALL_ORDER:
        for tier, filled in (("test", True), ("val", False)):
            m = (d.wall_type == w) & (d.tier == tier)
            ax.scatter(d.loc[m, "n_train_shells"] + jit[m], d.loc[m, "image_accuracy"], s=24, facecolor=WALL_COLOR[w] if filled else "white", edgecolor=WALL_COLOR[w], lw=0.9, label=f"{w.capitalize()}, {tier}", alpha=0.9)
    grp = d.groupby("n_train_shells").agg(img=("image_accuracy", "mean"), shell=("majority_correct", "mean"), n=("species", "size"))
    ax.plot(grp.index, grp.img, "k_", ms=18, mew=1.5)
    for k, r in grp.iterrows():
        ax.text(k, 1.04, f"n = {int(r.n)}\nshell {r.shell:.2f}", ha="center", fontsize=5.5, color="#333")
    ax.set_xticks(sorted(d.n_train_shells.unique()))
    ax.set_xlabel("Training individuals of the species")
    ax.set_ylabel("Fraction of the shell's 12 images correct")
    ax.set_ylim(-0.03, 1.15)
    ax.legend(frameon=False, fontsize=5, loc="center right", bbox_to_anchor=(1.0, 0.45), ncol=1)
    low = d[d.image_accuracy < 0.34].sort_values(["n_train_shells", "image_accuracy"]).reset_index(drop=True)
    stack: dict = {}
    for _, r in low.iterrows():
        short = r.display_name.split()[0][0] + ". " + " ".join(r.display_name.split()[1:])
        key = (r.n_train_shells, round(r.image_accuracy, 2))
        k = stack.get(key, 0)
        stack[key] = k + 1
        px = r.n_train_shells + jit[d.index[(d.species == r.species) & (d.specimen_id == r.specimen_id)][0]]
        ax.annotate(short, (px, r.image_accuracy), textcoords="offset points", xytext=(8, 2 + 8 * k), ha="left", va="bottom", fontsize=5, style="italic",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888", shrinkA=0, shrinkB=1.5))
    # right: image- and shell-level accuracy by training individuals, with Wilson intervals
    ax = axes[1]
    x = np.arange(len(grp))
    img_n = d.groupby("n_train_shells").apply(lambda g: g.n_images.sum())
    img_k = d.groupby("n_train_shells").apply(lambda g: g.n_correct_images.sum())
    shell_k = d.groupby("n_train_shells").majority_correct.sum()

    def wilson(k, n, z=1.96):
        p_ = k / n
        den = 1 + z**2 / n
        c = (p_ + z**2 / (2 * n)) / den
        h = z * np.sqrt(p_ * (1 - p_) / n + z**2 / (4 * n**2)) / den
        return c - h, c + h

    for off, k, n, lab, col in ((-0.18, img_k, img_n, "Images", "#888"), (0.18, shell_k, grp.n, "Shells (majority of 12)", "#c0392b")):
        vals = (k / n).to_numpy()
        lo, hi = wilson(k.to_numpy(), n.to_numpy())
        ax.bar(x + off, vals, 0.36, color=col, label=lab)
        ax.errorbar(x + off, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(k)}\n({int(r.n)} shells)" for k, r in grp.iterrows()], fontsize=7)
    ax.set_xlabel("Training individuals of the species")
    ax.set_ylabel("Accuracy on held-out shells")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel(axes[0], "A", x=-0.14)
    panel(axes[1], "B", x=-0.14)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rarity_f1.png", dpi=DPI)
    plt.close(fig)
    grp.to_csv(REP / arm / "rarity_by_training_shells.csv")
    print("rarity written")


def fig_ss_vs_ds() -> None:
    ss, ds = REP / "single_stream" / "test_specimens.csv", REP / "dual_stream" / "test_specimens.csv"
    if not (ss.exists() and ds.exists()):
        print("ss_vs_ds skipped")
        return
    a = pd.read_csv(ss).merge(pd.read_csv(ds), on=["species", "specimen_id"], suffixes=("_ss", "_ds"))
    t = table1().set_index("code_label")
    a["_w"] = a.species.map(lambda s: WALL_ORDER.index(t.loc[s, "wall_type"]))
    a = a.sort_values(["_w", "image_accuracy_ss"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(a))
    ax.bar(x - 0.2, a.image_accuracy_ss, 0.4, label="SS-MV (fraction of 12 views correct)", color="#c0392b")
    ax.bar(x + 0.2, a.image_accuracy_ds, 0.4, label="DS-MV", color="#4C72B0")
    for i, r in a.iterrows():
        ax.text(i - 0.2, 1.02, "✓" if r.majority_correct_ss else "✗", ha="center", fontsize=6, color="#c0392b")
        ax.text(i + 0.2, 1.02, "✓" if r.majority_correct_ds else "✗", ha="center", fontsize=6, color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([italic(t.loc[s, "display_name"]) for s in a.species], rotation=90, fontsize=5)
    for lab, s in zip(ax.get_xticklabels(), a.species, strict=True):
        lab.set_color(WALL_COLOR[t.loc[s, "wall_type"]])
    ax.text(len(a) - 0.5, 1.085, "✓ / ✗ : majority vote over the shell correct / wrong", ha="right", va="center", fontsize=5, color="#555")
    ax.set_ylabel("Per-shell image accuracy")
    ax.set_ylim(0, 1.13)
    ax.set_xlim(-0.7, len(a) - 0.4)
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ss_vs_ds.png", dpi=DPI)
    plt.close(fig)
    print("ss_vs_ds written")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    fig1()
    fig2()
    fig_s3_per_species()
    fig7()
    fig3()
    fig4()
    fig5()
    fig6()
    fig8()
    fig_rarity()
    fig_ss_vs_ds()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
