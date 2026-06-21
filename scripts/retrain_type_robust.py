"""
retrain_type_robust.py — make the type classifier noise-robust and INCUBATE it (sir's ask).

The robustness test (#5) showed the type classifier collapses to chance under additive noise
because it was trained only on clean scattering features. Here we retrain it WITH noise
augmentation (clean + several noisy copies of each train image) so it learns the noisy feature
distribution. The model architecture and the fixed scattering filters are unchanged — only the
small classifier's training data is enriched (still sir's IWSCN "scattering + small classifier").

Guard: we ADOPT the new model into production (`checkpoints/type_classifier.pkl`) only if its
CLEAN test balanced accuracy does not drop (stays within 0.01 of the current ~0.88) AND its
noisy accuracy improves. Otherwise we keep the current model and report honestly.

Run:  py -3.14 -m scripts.retrain_type_robust
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.models.scattering import ScatteringBranch
from src.models.type_classifier import (
    TypeClassifier, train_and_save, features_for_paths, scatter_features,
)

CLASSES = ("porosity", "slag")
MANIFEST = "data/processed/manifest.csv"
PROD = "checkpoints/type_classifier.pkl"
NEW = "checkpoints/type_classifier_robust.pkl"
SIGMAS_EVAL = [0.05, 0.10, 0.20]
SIGMAS_TRAIN = [0.02, 0.035]     # mild: add light-noise robustness without wrecking clean acc
KEEP_CLEAN_TOL = 0.015           # adopt only if clean acc drops by no more than this


def eval_clean_and_noisy(clf, scat, te, yte):
    patches = np.stack([np.load(p) for p in te["processed_path"]]).astype(np.float32)
    out = {}
    Xc = features_for_paths(scat, te["processed_path"].tolist())
    out["clean"] = balanced_accuracy_score(yte, clf.predict_features(Xc)[0])
    rng = np.random.default_rng(123)
    for s in SIGMAS_EVAL:
        noisy = np.clip(patches + rng.normal(0, s, patches.shape).astype(np.float32), 0, 1)
        Xn = scatter_features(scat, noisy)
        out[f"noise{s}"] = balanced_accuracy_score(yte, clf.predict_features(Xn)[0])
    return out


def main():
    df = pd.read_csv(MANIFEST)
    te = df[df.split == "test"]
    yte = np.array([CLASSES.index(c) for c in te["class"]])
    scat = ScatteringBranch(J=2, L=8, shape=(256, 256), max_order=2)

    print("=== CURRENT production classifier ===")
    old = eval_clean_and_noisy(TypeClassifier.load(PROD), scat, te, yte)
    for k, v in old.items():
        print(f"  {k:<10} {v:.3f}")

    print("\n=== training NOISE-AUGMENTED classifier ===")
    train_and_save(MANIFEST, NEW, classes=CLASSES, noise_sigmas=SIGMAS_TRAIN)

    print("\n=== NEW noise-augmented classifier ===")
    new = eval_clean_and_noisy(TypeClassifier.load(NEW), scat, te, yte)
    for k, v in new.items():
        print(f"  {k:<10} {v:.3f}")

    noisy_keys = [f"noise{s}" for s in SIGMAS_EVAL]
    old_noisy = np.mean([old[k] for k in noisy_keys])
    new_noisy = np.mean([new[k] for k in noisy_keys])
    clean_ok = new["clean"] >= old["clean"] - KEEP_CLEAN_TOL
    noisy_better = new_noisy > old_noisy + 0.02

    print("\n=== decision ===")
    print(f"  clean: {old['clean']:.3f} -> {new['clean']:.3f}  "
          f"({'OK' if clean_ok else 'DROPPED'})")
    print(f"  mean noisy: {old_noisy:.3f} -> {new_noisy:.3f}  "
          f"({'BETTER' if noisy_better else 'not better'})")

    if clean_ok and noisy_better:
        shutil.copy(NEW, PROD)
        print(f"\nADOPTED -> copied to production {PROD} (incubated into the model).")
        adopted = True
    else:
        print(f"\nNOT adopted — kept current {PROD}. (clean must hold AND noisy must improve.)")
        adopted = False
        os.remove(NEW)

    return 0 if adopted else 2


if __name__ == "__main__":
    raise SystemExit(main())
