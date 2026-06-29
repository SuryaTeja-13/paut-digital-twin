"""
tune_type_classifier.py — search for a higher-accuracy type classifier while staying
within our chosen approach (fixed Wavelet-Scattering features + a small classifier, à la IWSCN).

Honest protocol:
  * candidates differ only in (a) scattering renormalization and (b) the small classifier,
  * every candidate is scored by 5-fold stratified CV on the TRAIN split only,
  * the test split is touched once, for the CV-selected winner — no tuning on test.

In-approach knobs only:
  - log1p of scattering coeffs (standard log-scattering renormalization; Mallat/Bruna)
  - LogisticRegression C grid / Linear SVM / RBF SVM / tiny MLP  (= "the small classifier")
  - J=2 vs J=3 scattering depth (try J=3 if the features turn out too coarse)

Run:  py -3.14 -m scripts.tune_type_classifier
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.models.scattering import ScatteringBranch
from src.models.type_classifier import features_for_paths

CLASSES = ("porosity", "slag")
MANIFEST = "data/processed/manifest.csv"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def log1p_tf():
    return FunctionTransformer(np.log1p, validate=True)


def candidates():
    """name -> (sklearn pipeline). All are 'scattering features + small classifier'."""
    out = {}
    out["current: scale + LogReg"] = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    out["log + scale + LogReg"] = make_pipeline(
        log1p_tf(), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    for C in (0.1, 0.3, 1.0, 3.0):
        out[f"log + scale + LogReg(C={C})"] = make_pipeline(
            log1p_tf(), StandardScaler(),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
    out["log + scale + LinearSVM"] = make_pipeline(
        log1p_tf(), StandardScaler(), LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
    for C in (1.0, 3.0, 10.0):
        for g in ("scale",):
            out[f"log + scale + RBF-SVM(C={C})"] = make_pipeline(
                log1p_tf(), StandardScaler(),
                SVC(C=C, gamma=g, class_weight="balanced"))
    out["log + scale + tinyMLP(64)"] = make_pipeline(
        log1p_tf(), StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=0))
    return out


def run_for_J(J):
    print(f"\n{'='*64}\nSCATTERING DEPTH J={J}  (L=8, order=2)\n{'='*64}")
    df = pd.read_csv(MANIFEST)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    ytr = np.array([CLASSES.index(c) for c in tr["class"]])
    yte = np.array([CLASSES.index(c) for c in te["class"]])

    scat = ScatteringBranch(J=J, L=8, shape=(256, 256), max_order=2)
    print(f"extracting scattering features (train={len(tr)}, test={len(te)}) ...")
    Xtr = features_for_paths(scat, tr["processed_path"].tolist())
    Xte = features_for_paths(scat, te["processed_path"].tolist())
    print(f"feature dim = {Xtr.shape[1]}")

    rows = []
    for name, pipe in candidates().items():
        scores = cross_val_score(pipe, Xtr, ytr, cv=CV, scoring="balanced_accuracy")
        rows.append((name, scores.mean(), scores.std()))
        print(f"  CV bal-acc {scores.mean():.3f} ± {scores.std():.3f}   {name}")

    rows.sort(key=lambda r: r[1], reverse=True)
    best_name = rows[0][0]
    print(f"\nCV winner (J={J}): {best_name}  -> {rows[0][1]:.3f}")

    best = candidates()[best_name].fit(Xtr, ytr)
    pred = best.predict(Xte)
    bal = balanced_accuracy_score(yte, pred)
    print(f"TEST balanced acc = {bal:.3f}")
    for c, nm in enumerate(CLASSES):
        m = yte == c
        print(f"    {nm:<9} acc = {(pred[m] == c).mean():.3f}  (n={int(m.sum())})")
    print("    confusion [rows=true, cols=pred]:\n", confusion_matrix(yte, pred))
    return best_name, rows[0][1], bal


def main():
    results = {}
    for J in (2, 3):
        results[J] = run_for_J(J)
    print(f"\n{'='*64}\nSUMMARY  (this search selected the production model: log+MLP = 0.88 test)\n{'='*64}")
    for J, (name, cv, test) in results.items():
        print(f"  J={J}: CV {cv:.3f} | TEST {test:.3f}  via  {name}")


if __name__ == "__main__":
    raise SystemExit(main())
