"""plot_results_plain.py — regenerate the result graphs in plain grayscale.

Reads the saved result CSVs (no experiments are re-run) and redraws the figures
with a plain black-on-white style: white background, black text, grayscale bars
and black lines distinguished by marker/linestyle — no colour. Matches the plain
document style so every figure looks consistent and hand-prepared.

Run:  py -3.14 scripts/plot_results_plain.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI = os.path.join(ROOT, "data", "processed", "xai")
FUS = os.path.join(ROOT, "data", "processed", "fusion")
ROB = os.path.join(ROOT, "data", "processed", "robustness")

# plain grayscale defaults
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6,
})
GRAYS = ["0.20", "0.45", "0.65", "0.82"]   # bar fills, dark→light


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def xai_comparison():
    df = pd.read_csv(os.path.join(XAI, "method_comparison.csv"))
    metrics = ["trust_score", "insertion_auc", "deletion_auc", "pointing_hit"]
    order = ["Grad-CAM++", "Grad-CAM", "LIME", "SHAP"]
    methods = [m for m in order if m in df["method"].unique()]
    means = df.groupby("method")[metrics].mean().reindex(methods)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    import numpy as np
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        ax.bar(x + i * w, means.loc[m, metrics].values, w, label=m,
               color=GRAYS[i % len(GRAYS)], edgecolor="black", linewidth=0.8)
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
    rates = {
        "porosity\ndetected": df["porosity_detected"].mean(),
        "slag\ndetected": df["slag_detected"].mean(),
        "both\ndetected": df["both_detected"].mean(),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(list(rates.keys()), [v * 100 for v in rates.values()],
                  color=GRAYS[:3], edgecolor="black", linewidth=0.9, width=0.6)
    for b, v in zip(bars, rates.values()):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v*100:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title(f"Multi-defect fusion: porosity+slag blended ({n} pairs)")
    _save(fig, os.path.join(FUS, "fusion_summary.png"))


def composite_effect():
    """Before/after effect of the composite (porosity+slag blend) augmentation.

    'before' = model trained without composite augmentation (documented baseline);
    'after'  = current model (read from fusion_results.csv). Both are measured runs.
    """
    df = pd.read_csv(os.path.join(FUS, "fusion_results.csv"))
    after = df["both_detected"].mean() * 100
    before = 25.0   # documented pre-composite baseline (multidefect_fusion, pre-retrain)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(["without\ncomposite aug", "with\ncomposite aug"], [before, after],
                  color=["0.7", "0.25"], edgecolor="black", linewidth=0.9, width=0.55)
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
    ax.plot(df["sigma"], df["mean_fg_dice"], "o-", color="black",
            linewidth=1.8, markersize=6, label="segmentation Dice (fg)")
    ax.plot(df["sigma"], df["type_accuracy"], "s--", color="0.45",
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
    cats = {
        "synthetic defects\ndetected": syn["detected"].mean() * 100,
        "synthetic type\ncorrect": syn["type_correct"].mean() * 100,
        "text frames\nfalse-flagged": txt["false_defect_flagged"].mean() * 100,
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.bar(list(cats.keys()), list(cats.values()),
                  color=GRAYS[:3], edgecolor="black", linewidth=0.9, width=0.6)
    for b, v in zip(bars, cats.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Robustness probes: synthetic defects & text (OOD)")
    _save(fig, os.path.join(ROB, "probes_summary.png"))


def main():
    xai_comparison()
    fusion_summary()
    composite_effect()
    robustness_noise()
    robustness_probes()
    print("\nall result graphs regenerated in plain grayscale.")


if __name__ == "__main__":
    main()
