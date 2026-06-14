"""
check_separability.py — quick test of whether porosity vs slag is separable.

Extracts the global Wavelet-Scattering feature vector (mean+std over space) for
each image and fits a simple Logistic Regression (the reference paper's small-
sample recipe), evaluating on the val split. Reports BALANCED accuracy and a
confusion matrix because the corrected dataset is imbalanced (269 / 525).

Run AFTER build_dataset.py:
    py -3.14 scripts/check_separability.py
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from src.models.scattering import ScatteringBranch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def scatter_features(paths, scat, batch=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            arr = np.stack([np.load(p) for p in paths[i:i + batch]])
            x = torch.from_numpy(arr).float().unsqueeze(1)
            s = scat(x)
            v = torch.cat([s.mean(dim=(2, 3)), s.std(dim=(2, 3))], dim=1)
            out.append(v.numpy())
    return np.concatenate(out)


def main():
    df = pd.read_csv("data/processed/manifest.csv")
    classes = sorted(df["class"].unique())
    print("class counts:\n", df.groupby(["split", "class"]).size())

    scat = ScatteringBranch(J=2, L=8, shape=(256, 256), max_order=2)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ytr = (tr["class"] == "slag").astype(int).values
    yva = (va["class"] == "slag").astype(int).values

    print("\nextracting scattering features ...")
    Xtr = scatter_features(tr["processed_path"].tolist(), scat)
    Xva = scatter_features(va["processed_path"].tolist(), scat)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    pred = clf.predict(Xva)
    print(f"\ntrain acc          : {clf.score(Xtr, ytr):.3f}")
    print(f"val acc            : {clf.score(Xva, yva):.3f}")
    print(f"val BALANCED acc   : {balanced_accuracy_score(yva, pred):.3f}  "
          f"(0.5 = chance)")
    print("val confusion matrix [rows=true por/slag, cols=pred por/slag]:")
    print(confusion_matrix(yva, pred))
    print("\nInterpretation: balanced acc >> 0.5 means the classes ARE separable "
          "and a fixed/retrained model should classify type; ~0.5 means still not.")


if __name__ == "__main__":
    main()
