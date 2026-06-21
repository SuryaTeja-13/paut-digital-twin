"""Render the SCN-Attention U-Net architecture as a publication-quality figure.

All block names and tensor dimensions are the verified values from
``docs/model_spec.md`` (read from the instantiated model). Run:

    py -3.14 scripts/architecture_figure.py

Outputs ``docs/figures/architecture.png`` (300 dpi) and ``.pdf`` (vector).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt

# ----- palette -------------------------------------------------------------
C_ENC = "#2E5E8C"      # encoder (blue)
C_DEC = "#2E8C5E"      # decoder (green)
C_BOTTLE = "#6A3D9A"   # bottleneck / latent (purple)
C_SCAT = "#B0B0B0"     # scattering (grey, frozen)
C_FUSE = "#E08A2B"     # fusion (orange)
C_HEAD = "#C0392B"     # heads (red)
C_INPUT = "#34495E"    # input (dark slate)
TXT = "#FFFFFF"
EDGE = "#1A1A1A"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, color, title, sub, dashed=False, fs_t=9.5, fs_s=8):
    style = "round,pad=0.02,rounding_size=0.06"
    fb = FancyBboxPatch(
        (x, y), w, h, boxstyle=style, linewidth=1.6,
        edgecolor=EDGE, facecolor=color,
        linestyle="--" if dashed else "-", zorder=3,
    )
    ax.add_patch(fb)
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + h * 0.16, title, ha="center", va="center",
                color=TXT, fontsize=fs_t, fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.20, sub, ha="center", va="center",
                color=TXT, fontsize=fs_s, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                color=TXT, fontsize=fs_t, fontweight="bold", zorder=4)
    return (cx, cy)


def arrow(ax, p1, p2, color=EDGE, style="-", lw=1.6, rad=0.0):
    ar = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=13,
        linewidth=lw, color=color, linestyle=style,
        connectionstyle=f"arc3,rad={rad}", zorder=2,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(ar)


def main():
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    BW, BH = 1.9, 0.86  # box width / height

    # ---- encoder column (x ~ 2.6), going down ----
    xe = 2.6
    enc_y = [7.2, 5.95, 4.7, 3.45]
    enc = [
        ("enc1", "32 x 256 x 256"),
        ("enc2", "64 x 128 x 128"),
        ("enc3", "128 x 64 x 64"),
        ("enc4", "256 x 32 x 32"),
    ]
    enc_c = []
    for (t, s), y in zip(enc, enc_y):
        enc_c.append(box(ax, xe, y, BW, BH, C_ENC, t, s))

    # input on top
    in_c = box(ax, xe, 8.35, BW, BH, C_INPUT, "PAUT patch", "1 x 256 x 256")
    arrow(ax, (in_c[0], 8.35), (enc_c[0][0], enc_c[0][1] + BH / 2))

    # downsample arrows between encoders
    for a, b in zip(enc_c[:-1], enc_c[1:]):
        arrow(ax, (a[0], a[1] - BH / 2), (b[0], b[1] + BH / 2))
        ax.text(a[0] + 0.18, (a[1] + b[1]) / 2, "MaxPool 2", ha="left",
                va="center", fontsize=7, color="#555", style="italic")

    # ---- bottleneck (bottom center) ----
    bx, by = 5.55, 2.0
    bot_c = box(ax, bx, by, 2.4, BH, C_BOTTLE,
                "BOTTLENECK (latent)", "512 x 16 x 16")
    arrow(ax, (enc_c[3][0], enc_c[3][1] - BH / 2), (bot_c[0] - 0.5, by + BH / 2), rad=-0.2)

    # ---- decoder column (x ~ 8.7), going up ----
    xd = 8.9
    dec_y = [3.45, 4.7, 5.95, 7.2]
    dec = [
        ("up4", "256 x 32 x 32"),
        ("up3", "128 x 64 x 64"),
        ("up2", "64 x 128 x 128"),
        ("up1", "32 x 256 x 256"),
    ]
    dec_c = []
    for (t, s), y in zip(dec, dec_y):
        dec_c.append(box(ax, xd, y, BW, BH, C_DEC, t, s))

    arrow(ax, (bot_c[0] + 1.2 + 0.0, by + BH / 2), (dec_c[0][0], dec_c[0][1] - BH / 2), rad=-0.2)
    for a, b in zip(dec_c[:-1], dec_c[1:]):
        arrow(ax, (a[0], a[1] + BH / 2), (b[0], b[1] - BH / 2))
        ax.text(a[0] + 0.18, (a[1] + b[1]) / 2, "Up-conv x2", ha="left",
                va="center", fontsize=7, color="#555", style="italic")

    # ---- skip connections with Attention Gates ----
    for e, d in zip(enc_c, dec_c):
        agx = (e[0] + d[0]) / 2
        agy = e[1]
        cir = mpatches.Circle((agx, agy), 0.20, facecolor="#F1C40F",
                              edgecolor=EDGE, linewidth=1.3, zorder=4)
        ax.add_patch(cir)
        ax.text(agx, agy, "AG", ha="center", va="center", fontsize=7,
                fontweight="bold", zorder=5)
        arrow(ax, (e[0] + BW / 2, e[1]), (agx - 0.2, agy), style="--", lw=1.2, color="#777")
        arrow(ax, (agx + 0.2, agy), (d[0] - BW / 2, d[1]), style="--", lw=1.2, color="#777")

    # ---- scattering branch (far left, frozen) ----
    sx, sy = 0.15, 5.2
    scat_c = box(ax, sx, sy, 2.15, 1.2, C_SCAT,
                 "Wavelet Scattering", "J=2 L=8 ord=2\n81 x 64 x 64  (0 params)",
                 dashed=True, fs_t=9, fs_s=7)
    ax.text(sx + 1.07, sy + 1.35, "FIXED PHYSICS PRIOR", ha="center",
            fontsize=7.5, fontweight="bold", color="#444")

    # fusion arrows into each encoder + bottleneck
    fuse_targets = enc_c + [(bot_c[0] - 1.2, bot_c[1])]
    for t in fuse_targets:
        arrow(ax, (sx + 2.15, sy + 0.6), (t[0] - BW / 2 - 0.02, t[1]),
              color=C_FUSE, lw=1.1, rad=0.12)
    ax.text(sx + 1.07, sy - 0.35, "SCN-Fusion + CBAM\n(1x1 conv per level)",
            ha="center", fontsize=7, color=C_FUSE, fontweight="bold")

    # ---- heads (right) ----
    # segmentation head
    seg_c = box(ax, 11.55, 7.2, 2.9, BH, C_HEAD,
                "Segmentation head", "Conv 1x1 -> 3 x 256 x 256")
    arrow(ax, (dec_c[3][0] + BW / 2, dec_c[3][1]), (11.55, seg_c[1]))
    ax.text(13.0, 6.62, "background / porosity / slag", ha="center",
            fontsize=7, style="italic", color="#444")

    # classifier head
    cls_c = box(ax, 11.55, 3.0, 2.9, 1.1, C_HEAD,
                "Type classifier head",
                "concat(512 + 81)=593\nLinear 128 -> 2 classes")
    # latent -> classifier
    arrow(ax, (bot_c[0] + 1.2, bot_c[1]), (11.7, cls_c[1] - 0.2), rad=-0.35)
    ax.text(9.6, 2.15, "GAP latent (512)", ha="center", fontsize=7, color=C_BOTTLE)
    # scattering vector -> classifier
    arrow(ax, (sx + 2.0, sy), (11.55, cls_c[1] + 0.2), color=C_FUSE, lw=1.0, rad=-0.45)
    ax.text(6.6, 1.15, "global scattering vector (81)", ha="center",
            fontsize=7, color=C_FUSE)
    ax.text(13.0, 2.35, "porosity / slag", ha="center", fontsize=7,
            style="italic", color="#444")

    # ---- title + footer ----
    ax.text(7.5, 9.35, "SCN-Attention U-Net  —  PAUT Explainable-AI Digital Twin",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(7.5, 0.30,
            "8,798,151 trainable parameters  |  scattering filters add 0 (fixed buffers)  "
            "|  two heads share one encoder",
            ha="center", fontsize=8.5, color="#333")

    # ---- legend ----
    legend = [
        (C_ENC, "Encoder"), (C_BOTTLE, "Bottleneck / latent"),
        (C_DEC, "Decoder"), (C_SCAT, "Scattering (frozen)"),
        (C_FUSE, "SCN-Fusion + CBAM"), (C_HEAD, "Output heads"),
    ]
    lx = 0.2
    for col, lab in legend:
        ax.add_patch(mpatches.Rectangle((lx, 0.05), 0.28, 0.28,
                     facecolor=col, edgecolor=EDGE, linewidth=1))
        ax.text(lx + 0.36, 0.19, lab, va="center", fontsize=7.5)
        lx += len(lab) * 0.085 + 0.7

    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "architecture.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out / "architecture.pdf", bbox_inches="tight", facecolor="white")
    print("wrote", out / "architecture.png", "and .pdf")


if __name__ == "__main__":
    main()
