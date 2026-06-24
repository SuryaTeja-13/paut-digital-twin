"""plot_results.py — regenerate every result graph in colour.

Reads the saved result CSVs (no experiments are re-run) and redraws the figures
with coloured bars and lines on a clean white background with black text/axes —
the data series carry colour, the document text stays black-on-white. Matches the
colour style used in the report so every figure is consistent.

Run:  py -3.14 scripts/plot_results.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI = os.path.join(ROOT, "data", "processed", "xai")
FUS = os.path.join(ROOT, "data", "processed", "fusion")
ROB = os.path.join(ROOT, "data", "processed", "robustness")
ABL = os.path.join(ROOT, "data", "processed", "ablation")

# black text/axes (document content), coloured data series (tab10 — matches report)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
})
BLUE, RED, GREEN, ORANGE, NAVY = "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#1f4e79"
PALETTE = [BLUE, RED, GREEN, ORANGE]   # Grad-CAM++, Grad-CAM, LIME, SHAP


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def xai_comparison():
    df = pd.read_csv(os.path.join(XAI, "method_comparison.csv"))
    metrics = ["trust_score", "insertion_auc", "deletion_auc", "pointing_hit"]
    order = ["Grad-CAM++", "Grad-CAM", "LIME", "SHAP"]
    methods = [m for m in order if m in df["method"].unique()]
    means = df.groupby("method")[metrics].mean().reindex(methods)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        ax.bar(x + i * w, means.loc[m, metrics].values, w, label=m,
               color=PALETTE[i % len(PALETTE)], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x + w * (len(methods) - 1) / 2)
    ax.set_xticklabels(["trust score", "insertion AUC", "deletion AUC\n(lower=better)",
                        "pointing-game"], fontsize=9)
    ax.set_ylabel("score")
    ax.set_title("XAI method comparison (mean over test images)")
    ax.legend(frameon=True, edgecolor="black", facecolor="white")
    ax.set_ylim(0, 1.05)
    _save(fig, os.path.join(XAI, "method_comparison.png"))


def fusion_summary():
    df = pd.read_csv(os.path.join(FUS, "fusion_results.csv"))
    n = len(df)
    rates = {"porosity\ndetected": df["porosity_detected"].mean(),
             "slag\ndetected": df["slag_detected"].mean(),
             "both\ndetected": df["both_detected"].mean()}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(list(rates.keys()), [v * 100 for v in rates.values()],
                  color=[BLUE, ORANGE, GREEN], edgecolor="black", linewidth=0.9, width=0.6)
    for b, v in zip(bars, rates.values()):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"Multi-defect fusion: porosity+slag blended ({n} pairs)")
    _save(fig, os.path.join(FUS, "fusion_summary.png"))


def composite_effect():
    df = pd.read_csv(os.path.join(FUS, "fusion_results.csv"))
    after = df["both_detected"].mean() * 100
    before = 25.0   # documented pre-composite baseline (multidefect_fusion, pre-retrain)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["without\ncomposite aug", "with\ncomposite aug"], [before, after],
                  color=[ORANGE, GREEN], edgecolor="black", linewidth=0.9, width=0.55)
    for b, v in zip(bars, [before, after]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("both defects detected (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Composite augmentation: multi-defect detection")
    _save(fig, os.path.join(FUS, "composite_effect.png"))


def robustness_noise():
    df = pd.read_csv(os.path.join(ROB, "noise_curve.csv")).sort_values("sigma")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(df["sigma"], df["mean_fg_dice"], "o-", color=BLUE,
            linewidth=1.8, markersize=6, label="segmentation Dice (fg)")
    ax.plot(df["sigma"], df["type_accuracy"], "s--", color=ORANGE,
            linewidth=1.6, markersize=6, label="type accuracy")
    ax.set_xlabel("added Gaussian noise σ")
    ax.set_ylabel("score")
    ax.set_title("Robustness to additive noise")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, edgecolor="black", facecolor="white")
    _save(fig, os.path.join(ROB, "noise_curve.png"))


def robustness_probes():
    syn = pd.read_csv(os.path.join(ROB, "synthetic.csv"))
    txt = pd.read_csv(os.path.join(ROB, "text_probe.csv"))
    cats = {"synthetic defects\ndetected": syn["detected"].mean() * 100,
            "synthetic type\ncorrect": syn["type_correct"].mean() * 100,
            "text frames\nfalse-flagged": txt["false_defect_flagged"].mean() * 100}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(list(cats.keys()), list(cats.values()),
                  color=[GREEN, BLUE, RED], edgecolor="black", linewidth=0.9, width=0.6)
    for b, v in zip(bars, cats.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Robustness probes: synthetic defects & text (OOD)")
    _save(fig, os.path.join(ROB, "probes_summary.png"))


def _ablation_curve(csv_name, png_name, title_suffix):
    path = os.path.join(ABL, csv_name)
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for use_scat, label, style, col in [(True, "SCN-Attention U-Net", "o-", NAVY),
                                        (False, "plain U-Net (baseline)", "s--", RED)]:
        sub = df[df["use_scattering"] == use_scat].sort_values("data_fraction")
        if sub.empty:
            continue
        fr = sub["data_fraction"] * 100
        ax[0].plot(fr, sub["best_val_dice_fg"], style, color=col, lw=1.8,
                   markersize=6, label=label)
        ax[1].plot(fr, sub["best_val_cls_acc"], style, color=col, lw=1.8,
                   markersize=6, label=label)
    ax[0].set_title(f"Segmentation: val dice_fg vs training-data size{title_suffix}")
    ax[1].set_title(f"Classification: val accuracy vs training-data size{title_suffix}")
    for a in ax:
        a.set_xlabel("training data used (%)")
        a.legend(frameon=True, edgecolor="black", facecolor="white")
    ax[0].set_ylabel("val dice_fg")
    ax[1].set_ylabel("val cls acc")
    _save(fig, os.path.join(ABL, png_name))


def main():
    xai_comparison()
    fusion_summary()
    composite_effect()
    robustness_noise()
    robustness_probes()
    _ablation_curve("ablation_results.csv", "ablation_curve.png", "")
    _ablation_curve("ablation_results_with_aug.csv", "ablation_curve_with_aug.png",
                    " (with aug)")
    print("\nall result graphs regenerated in colour.")


if __name__ == "__main__":
    main()
