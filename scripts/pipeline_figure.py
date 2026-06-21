"""Render the full end-to-end project pipeline (Student 1 -> Student 4).

A single system-level figure covering every milestone: preprocessing and
pseudo-labels (S1), the SCN-Attention U-Net and characterization (S2),
explainable AI (S3), and the digital twin / dashboard (S4). Stage names map
to the README milestone table. Run:

    py -3.14 scripts/pipeline_figure.py

Outputs ``docs/figures/pipeline_overview.png`` (300 dpi) and ``.pdf``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

# student band colors (header) + lighter box fills
BANDS = [
    ("Student 1", "Preprocessing & labels", "#1F6FB2", "#D6E8F5"),
    ("Student 2", "Model & characterization", "#9A3DA6", "#ECDDF0"),
    ("Student 3", "Explainable AI", "#C77A0A", "#F7E7CC"),
    ("Student 4", "Digital twin & dashboard", "#1E8449", "#D6EFE0"),
]
EDGE = "#1A1A1A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, fill, edge, title, lines, fs_t=9.5, fs_l=7.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=3))
    cx = x + w / 2
    ax.text(cx, y + h - 0.30, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color="#111", zorder=4)
    for i, ln in enumerate(lines):
        ax.text(cx, y + h - 0.62 - i * 0.30, ln, ha="center", va="center",
                fontsize=fs_l, color="#222", zorder=4)
    return (x, y, w, h, cx, y + h / 2)


def arrow(ax, a, b, color=EDGE, lw=2.0):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=16, linewidth=lw,
        color=color, zorder=2, shrinkA=3, shrinkB=3))


def main():
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # ---- four student bands (colored background columns) ----
    band_x = [1.0, 4.7, 9.1, 12.4]
    band_w = [3.4, 4.1, 3.0, 3.4]
    band_y, band_h = 1.05, 5.55
    for (sname, srole, dark, light), bx, bw in zip(BANDS, band_x, band_w):
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, band_y), bw, band_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=light, edgecolor=dark, linewidth=1.8,
            linestyle="-", zorder=1, alpha=0.55))
        # band header
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, band_y + band_h + 0.12), bw, 0.62,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=dark, edgecolor=dark, linewidth=1.5, zorder=2))
        ax.text(bx + bw / 2, band_y + band_h + 0.55, sname,
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=4)
        ax.text(bx + bw / 2, band_y + band_h + 0.28, srole,
                ha="center", va="center", color="white",
                fontsize=7.8, zorder=4)

    BW, BH = 2.9, 1.45
    yC = 3.55  # vertical center for the box row inside the bands

    # ---- input ----
    inb = box(ax, 0.05, yC - 0.7, 0.95, 1.4, "#34495E", EDGE,
              "Input", ["Raw PAUT", "scan image"], fs_t=8.5, fs_l=7)
    ax.text(0.525, yC - 0.95, "(.jpg/.png)", ha="center", fontsize=6.5, color="#555")

    chain = []

    # ---- S1: two stacked stages ----
    s1d, s1l = BANDS[0][2], BANDS[0][3]
    a = box(ax, 1.15, yC + 0.15, 3.1, 1.3, s1l, s1d, "Preprocess",
            ["color->amplitude", "crop . normalize", "split + manifest"])
    b = box(ax, 1.15, yC - 1.55, 3.1, 1.3, s1l, s1d, "Pseudo-labels",
            ["Stage-1 masks", "(weak supervision)"])
    chain += [a, b]

    # ---- S2 ----
    s2d, s2l = BANDS[1][2], BANDS[1][3]
    c = box(ax, 4.85, yC + 0.15, 3.75, 1.3, s2l, s2d, "SCN-Attention U-Net",
            ["scattering prior + Attn U-Net", "seg (3-cls) + type (2-cls)", "8.8M params"])
    d = box(ax, 4.85, yC - 1.55, 3.75, 1.3, s2l, s2d, "Characterization",
            ["blob analysis . PCA", "size / orientation / severity"])
    chain += [c, d]

    # ---- S3 ----
    s3d, s3l = BANDS[2][2], BANDS[2][3]
    e = box(ax, 9.25, yC - 0.7, 2.7, 1.4, s3l, s3d, "Explainable AI",
            ["Grad-CAM++ . attention", "faithfulness -> trust score"])
    chain += [e]

    # ---- S4 ----
    s4d, s4l = BANDS[3][2], BANDS[3][3]
    f = box(ax, 12.55, yC - 0.7, 3.1, 1.4, s4l, s4d, "Digital Twin + Dashboard",
            ["health index . live feed", "batch + continuous mode"])
    chain += [f]

    # ---- output ----
    # (drawn as a tall summary box on far right handled by f's outputs text)

    # ---- arrows along the main flow ----
    arrow(ax, (inb[4] + 0.475, yC), (1.15, yC + 0.15 + 0.65))   # input -> preprocess
    arrow(ax, (inb[4] + 0.475, yC), (1.15, yC - 1.55 + 0.65), lw=1.2, color="#888")
    # preprocess -> pseudo (down within S1)
    arrow(ax, (a[4], a[1]), (b[4], b[1] + b[3]), color=s1d, lw=1.6)
    # S1 -> S2 (both rows)
    arrow(ax, (a[0] + BW + 0.2 - 0.0, a[5]), (c[0], c[5]))
    arrow(ax, (b[0] + 3.1, b[5]), (d[0], d[5]))
    # S2 -> S3 (merge into XAI)
    arrow(ax, (c[0] + 3.75, c[5]), (e[0], e[5] + 0.25))
    arrow(ax, (d[0] + 3.75, d[5]), (e[0], e[5] - 0.25))
    # S3 -> S4
    arrow(ax, (e[0] + 2.7, e[5]), (f[0], f[5]))

    # ---- outputs callout (below S4) ----
    ax.add_patch(FancyBboxPatch(
        (12.55, 1.15), 3.1, 1.15, boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="#FBFBFB", edgecolor=s4d, linewidth=1.4,
        linestyle="--", zorder=3))
    ax.text(14.1, 2.05, "Final outputs", ha="center", fontsize=8.5,
            fontweight="bold", color=s4d)
    ax.text(14.1, 1.55,
            "defect map . type . severity\nhealth %  .  trust score  .  report",
            ha="center", va="center", fontsize=7.3, color="#222")
    arrow(ax, (f[4], f[1]), (14.1, 2.30), color=s4d, lw=1.6)

    # ---- title & footer ----
    ax.text(8.0, 7.92, "PAUT Explainable-AI Digital Twin  —  End-to-End Pipeline",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(8.0, 0.45,
            "Each colored band is one team member's milestone(s); data flows left -> right "
            "from a raw PAUT scan to a trustworthy, explained inspection report.",
            ha="center", fontsize=8.3, color="#333")

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pipeline_overview.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out / "pipeline_overview.pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out / "pipeline_overview.png", "and .pdf")


if __name__ == "__main__":
    main()
