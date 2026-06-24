"""Render the SCN-Attention U-Net architecture as a clean, publication-quality figure.

Every block name and tensor dimension is the verified value from docs/model_spec.md.
Layout is a clean U: scattering prior (far left) -> encoder (down) -> bottleneck (bottom)
-> decoder (up) -> two heads (right). Skip connections are horizontal at matching levels
(no crossing); the scattering prior fuses into each encoder level via short arrows.

Run:  py -3.14 scripts/architecture_figure.py
Outputs docs/figures/architecture.png (300 dpi) and .pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

C_ENC = "#2E5E8C"
C_DEC = "#2E8C5E"
C_BOT = "#6A3D9A"
C_SCAT = "#9AA0A6"
C_HEAD = "#C0392B"
C_IN = "#34495E"
EDGE = "#1A1A1A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})


def box(ax, cx, cy, w, h, color, title, sub, dashed=False, tfs=10, sfs=8.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=1.6,
                 edgecolor=EDGE, facecolor=color, linestyle="--" if dashed else "-", zorder=3))
    ax.text(cx, cy + h * 0.17, title, ha="center", va="center", color="white",
            fontsize=tfs, fontweight="bold", zorder=4)
    if sub:
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center", color="white",
                fontsize=sfs, zorder=4)


def arr(ax, p1, p2, color=EDGE, style="-", lw=1.8, rad=0.0, scale=14):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=scale,
                 linewidth=lw, color=color, linestyle=style, zorder=2,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    BW, BH = 2.05, 0.92
    ex, dx = 4.2, 10.3                     # encoder / decoder column x-centres
    ys = [7.3, 6.05, 4.8, 3.55]            # four levels
    enc = [("enc1", "32 x 256 x 256"), ("enc2", "64 x 128 x 128"),
           ("enc3", "128 x 64 x 64"), ("enc4", "256 x 32 x 32")]
    dec = [("up1", "32 x 256 x 256"), ("up2", "64 x 128 x 128"),
           ("up3", "128 x 64 x 64"), ("up4", "256 x 32 x 32")]

    # input
    box(ax, ex, 8.5, BW, BH, C_IN, "PAUT patch", "1 x 256 x 256", tfs=9, sfs=7.6)
    arr(ax, (ex, 8.5 - BH / 2), (ex, ys[0] + BH / 2))

    # encoder column (down)
    for (t, s), y in zip(enc, ys):
        box(ax, ex, y, BW, BH, C_ENC, t, s)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arr(ax, (ex, y1 - BH / 2), (ex, y2 + BH / 2))
        ax.text(ex + 0.16, (y1 + y2) / 2, "MaxPool 2", ha="left", va="center",
                fontsize=7, style="italic", color="#555")

    # bottleneck (bottom centre)
    by = 2.15
    box(ax, 7.25, by, 2.6, BH, C_BOT, "BOTTLENECK (latent)", "512 x 16 x 16", tfs=9, sfs=8)
    arr(ax, (ex, ys[3] - BH / 2), (7.25 - 0.7, by + BH / 2), rad=-0.18)

    # decoder column (up)
    for (t, s), y in zip(dec[::-1], ys[::-1]):     # up4 at bottom .. up1 at top
        box(ax, dx, y, BW, BH, C_DEC, t, s)
    arr(ax, (7.25 + 0.7, by + BH / 2), (dx, ys[3] - BH / 2), rad=-0.18)
    for y1, y2 in zip(ys[::-1][:-1], ys[::-1][1:]):
        arr(ax, (dx, y1 + BH / 2), (dx, y2 - BH / 2))
        ax.text(dx + 0.16, (y1 + y2) / 2, "Up-conv x2", ha="left", va="center",
                fontsize=7, style="italic", color="#555")

    # skip connections with Attention Gates (clean horizontals, AG next to decoder)
    agx = dx - BW / 2 - 0.55
    for y in ys:
        ax.add_patch(mpatches.Circle((agx, y), 0.19, facecolor="#F1C40F",
                     edgecolor=EDGE, linewidth=1.2, zorder=4))
        ax.text(agx, y, "AG", ha="center", va="center", fontsize=6.8, fontweight="bold", zorder=5)
        arr(ax, (ex + BW / 2, y), (agx - 0.19, y), style="--", lw=1.1, color="#777", scale=11)
        arr(ax, (agx + 0.19, y), (dx - BW / 2, y), style="--", lw=1.1, color="#777", scale=11)
    ax.text((ex + dx) / 2, ys[0] + 0.42, "skip connections (attention-gated)",
            ha="center", fontsize=7.5, style="italic", color="#666")

    # scattering prior (far left) + short fusion arrows into each encoder level
    sx = 1.35
    ax.add_patch(FancyBboxPatch((sx - 1.0, ys[3] - BH / 2 - 0.1), 2.0, ys[0] - ys[3] + BH + 0.2,
                 boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=1.6, edgecolor=EDGE,
                 facecolor=C_SCAT, linestyle="--", zorder=3))
    ax.text(sx, ys[0] + 0.05, "Wavelet", ha="center", color="white", fontsize=9.5, fontweight="bold", zorder=4)
    ax.text(sx, ys[0] - 0.30, "Scattering", ha="center", color="white", fontsize=9.5, fontweight="bold", zorder=4)
    ax.text(sx, 5.05, "J=2, L=8", ha="center", color="white", fontsize=8, zorder=4)
    ax.text(sx, 4.72, "81 x 64 x 64", ha="center", color="white", fontsize=8, zorder=4)
    ax.text(sx, 3.75, "fixed prior", ha="center", color="white", fontsize=8, style="italic", zorder=4)
    ax.text(sx, 3.45, "0 params", ha="center", color="white", fontsize=8, style="italic", zorder=4)
    for y in ys:
        arr(ax, (sx + 1.0, y), (ex - BW / 2, y), color="#E08A2B", lw=1.4, scale=12)
    fxc = (sx + 1.0 + ex - BW / 2) / 2
    ax.text(fxc, 5.55, "SCN-Fusion", ha="center", fontsize=6.6, color="#C77A0A",
            fontweight="bold", rotation=90)
    ax.text(fxc + 0.28, 5.55, "+ CBAM", ha="center", fontsize=6.6, color="#C77A0A",
            fontweight="bold", rotation=90)

    # heads (right)
    box(ax, dx + 3.0, ys[0], 2.5, BH, C_HEAD, "Segmentation head", "3 x 256 x 256", tfs=8.6, sfs=8)
    arr(ax, (dx + BW / 2, ys[0]), (dx + 3.0 - 2.5 / 2, ys[0]))
    ax.text(dx + 3.0, ys[0] - 0.62, "background / porosity / slag", ha="center",
            fontsize=7, style="italic", color="#444")

    box(ax, dx + 3.0, by + 0.15, 2.7, 1.05, C_HEAD, "Type classifier head",
        "concat(512+81)=593 -> 2", tfs=8.6, sfs=7.6)
    arr(ax, (7.25 + 1.3, by), (dx + 3.0 - 2.7 / 2, by + 0.15), color=C_BOT, lw=1.4, rad=-0.25)
    ax.text(9.5, 1.62, "pooled latent (512)", ha="center", fontsize=7, color=C_BOT)
    arr(ax, (sx, ys[3] - BH / 2 - 0.1), (dx + 3.0 - 2.7 / 2, by + 0.35),
        color="#E08A2B", lw=1.1, rad=-0.32, scale=11)
    ax.text(5.4, 0.95, "+ global scattering vector (81)", ha="center", fontsize=7, color="#C77A0A")

    ax.text(7.5, 9.15, "SCN-Attention U-Net  —  PAUT Explainable-AI Digital Twin",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(7.5, 0.32, "8,798,151 trainable parameters   |   scattering filters add 0 (fixed)"
            "   |   two heads share one encoder", ha="center", fontsize=8.5, color="#333")

    legend = [(C_ENC, "Encoder"), (C_BOT, "Bottleneck"), (C_DEC, "Decoder"),
              (C_SCAT, "Scattering (fixed)"), (C_HEAD, "Heads")]
    lx = 1.2
    for col, lab in legend:
        ax.add_patch(mpatches.Rectangle((lx, 0.55), 0.26, 0.26, facecolor=col,
                     edgecolor=EDGE, linewidth=1))
        ax.text(lx + 0.34, 0.68, lab, va="center", fontsize=7.5)
        lx += len(lab) * 0.092 + 0.9

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "architecture.pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out / "architecture.png")


if __name__ == "__main__":
    main()
