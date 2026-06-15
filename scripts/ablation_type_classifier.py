"""
ablation_type_classifier.py — data-ablation of the TYPE classifier (sir's §5.11 curriculum,
applied to the scattering-feature type classifier we improved to 0.88).

Trains the SAME in-approach classifier (log-scattering -> scale -> tiny MLP) on
100 / 50 / 25 / 10 % of the TRAIN split and scores balanced accuracy on the FULL
test split. Several random subsamples per fraction -> mean ± std (honest, low-noise).
Scattering features are fixed and extracted once, so the whole sweep runs on CPU in
minutes. Shows whether the fixed wavelet-scattering prior keeps the type signal
robust as data shrinks — the small-data point sir's approach is built to make.

Run:  py -3.14 -m scripts.ablation_type_classifier
Out:  data/processed/ablation/type_clf_ablation.{csv,png}
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score

from src.models.scattering import ScatteringBranch
from src.models.type_classifier import features_for_paths

CLASSES = ("porosity", "slag")
FRACTIONS = [1.0, 0.5, 0.25, 0.1]
N_SEEDS = 5
OUT_DIR = "data/processed/ablation"


def make_clf():
    return make_pipeline(
        FunctionTransformer(np.log1p, validate=True),
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=0),
    )


def stratified_subsample(y, frac, seed):
    """Indices for a class-stratified fraction of the rows."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(2, int(round(len(ci) * frac)))
        idx.extend(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.array(sorted(idx))


def main():
    df = pd.read_csv("data/processed/manifest.csv")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    ytr = np.array([CLASSES.index(c) for c in tr["class"]])
    yte = np.array([CLASSES.index(c) for c in te["class"]])

    scat = ScatteringBranch(J=2, L=8, shape=(256, 256), max_order=2)
    print(f"extracting scattering features once (train={len(tr)}, test={len(te)}) ...")
    Xtr = features_for_paths(scat, tr["processed_path"].tolist())
    Xte = features_for_paths(scat, te["processed_path"].tolist())

    rows = []
    for frac in FRACTIONS:
        bals, pors, slags = [], [], []
        seeds = [0] if frac >= 0.999 else list(range(N_SEEDS))
        n_used = 0
        for s in seeds:
            sel = stratified_subsample(ytr, frac, s) if frac < 0.999 else np.arange(len(ytr))
            n_used = len(sel)
            clf = make_clf().fit(Xtr[sel], ytr[sel])
            pred = clf.predict(Xte)
            bals.append(balanced_accuracy_score(yte, pred))
            pors.append((pred[yte == 0] == 0).mean())
            slags.append((pred[yte == 1] == 1).mean())
        row = {"data_fraction": frac, "n_train": n_used,
               "bal_acc_mean": round(float(np.mean(bals)), 4),
               "bal_acc_std": round(float(np.std(bals)), 4),
               "porosity_mean": round(float(np.mean(pors)), 4),
               "slag_mean": round(float(np.mean(slags)), 4)}
        rows.append(row)
        print(f"  {int(frac*100):>3d}%  n={n_used:>3d}  "
              f"balanced {row['bal_acc_mean']:.3f} ± {row['bal_acc_std']:.3f}  "
              f"(porosity {row['porosity_mean']:.3f}, slag {row['slag_mean']:.3f})")

    os.makedirs(OUT_DIR, exist_ok=True)
    res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "type_clf_ablation.csv")
    res.to_csv(csv_path, index=False)

    # plot: balanced accuracy vs training-data size, with ± std band
    x = [f * 100 for f in res["data_fraction"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, res["bal_acc_mean"], "o-", color="#1f4e79", lw=2,
            label="scattering type classifier (log-scattering + MLP)")
    ax.fill_between(x, res["bal_acc_mean"] - res["bal_acc_std"],
                    res["bal_acc_mean"] + res["bal_acc_std"], color="#1f4e79", alpha=0.15)
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance (0.50)")
    ax.set_xlabel("training data used (%)")
    ax.set_ylabel("test balanced accuracy")
    ax.set_title("Type-classification: balanced accuracy vs training-data size")
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "type_clf_ablation.png")
    fig.savefig(png_path, dpi=130)
    print(f"\nsaved -> {csv_path}\nsaved -> {png_path}")


if __name__ == "__main__":
    raise SystemExit(main())
