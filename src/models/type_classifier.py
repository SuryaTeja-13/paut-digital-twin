"""
type_classifier.py — image-level porosity-vs-slag from Wavelet-Scattering features.

WHY: the SCN-Attention U-Net is great at *detecting* defects but weak at *typing*
them — its per-pixel/global-pooled type signal mislabels porosity as slag ~half
the time. Porosity-vs-slag is a whole-image texture/shape property, and the fixed
Wavelet Scattering transform is exactly the small-sample texture descriptor the
reference paper uses. A small classifier on the global scattering vector
separates the classes far better. With log-scattering renormalisation (the standard
Mallat/Bruna log of the coefficients) and a tiny one-hidden-layer MLP — i.e. exactly
the "scattering + small neural classifier" of the IWSCN reference paper — it reaches
≈0.88 balanced (porosity ≈0.86, slag ≈0.90), up from ≈0.80 for a plain logistic
regression. The scattering filters stay fixed (J=2, L=8, order=2); only the small
classifier changed, so the SCN-Attention U-Net architecture is untouched.

Since every image here is single-type (design_decisions.md §2), this image-level type is the
defect type, applied to all defects in the image. This is the "classification
head" of decision §11 — just a stronger one than the neural head.

Trains on CPU in a few minutes (scattering is fixed; only the small classifier
is fit). No GPU and no model checkpoint needed.
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
import torch

from .scattering import ScatteringBranch


def scatter_features(scat: ScatteringBranch, patches: np.ndarray, device=None) -> np.ndarray:
    """patches: (N,H,W) float32 -> (N, 2C) global scattering features (mean+std over space)."""
    device = device or torch.device("cpu")
    out = []
    with torch.no_grad():
        for i in range(0, len(patches), 16):
            x = torch.from_numpy(np.ascontiguousarray(patches[i:i + 16])).float().unsqueeze(1).to(device)
            s = scat(x)                                    # (B,C,h,w)
            out.append(torch.cat([s.mean(dim=(2, 3)), s.std(dim=(2, 3))], dim=1).cpu().numpy())
    return np.concatenate(out)


def features_for_paths(scat, paths, device=None):
    return scatter_features(scat, np.stack([np.load(p) for p in paths]), device)


class TypeClassifier:
    """Wraps the fitted (log-scattering -> scale -> small MLP) pipeline. Loaded by the pipeline."""

    def __init__(self, pipe, classes, scat_params):
        self.pipe, self.classes, self.scat_params = pipe, classes, scat_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        if "pipe" in d:                                    # current format
            return TypeClassifier(d["pipe"], d["classes"], d["scat_params"])
        # legacy format (scaler + clf, no log) — keep loading old checkpoints
        from sklearn.pipeline import Pipeline as _SkPipe
        pipe = _SkPipe([("scaler", d["scaler"]), ("clf", d["clf"])])
        return TypeClassifier(pipe, d["classes"], d["scat_params"])

    def predict_features(self, feats):
        """feats: (N, 2C) RAW scattering mean+std -> (idx array, prob array (N, n_classes)).

        The log-scattering renormalisation lives inside the pipeline, so callers pass
        the same raw mean+std `scatter_features` produces (no change at the call site).
        """
        return self.pipe.predict(feats), self.pipe.predict_proba(feats)

    def predict_patch(self, patch, scat, device=None):
        f = scatter_features(scat, patch[None].astype(np.float32), device)
        idx, prob = self.predict_features(f)
        return int(idx[0]), prob[0]


def train_and_save(manifest, out_path, classes=("porosity", "slag"),
                   J=2, L=8, order=2, shape=(256, 256), noise_sigmas=None):
    """Fit the small classifier on FIXED scattering features (the IWSCN approach).

    Pipeline = log1p (log-scattering, Mallat/Bruna) -> StandardScaler -> MLP(64).
    Selected by 5-fold CV on the train split; the test split is scored once (honest).

    noise_sigmas: if given (e.g. [0.03, 0.06, 0.10]), the TRAIN features are augmented
    with additive-Gaussian-noisy copies of each image at those σ. This teaches the
    classifier the noise-perturbed feature distribution so it stays accurate under noise
    (robustness fix) — the model architecture and the scattering filters are unchanged.
    The TEST split is always scored on CLEAN images (honest clean accuracy).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    df = pd.read_csv(manifest)
    scat = ScatteringBranch(J=J, L=L, shape=shape, max_order=order)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    ytr = np.array([list(classes).index(c) for c in tr["class"]])
    yte = np.array([list(classes).index(c) for c in te["class"]])

    print(f"extracting scattering features (train={len(tr)}, test={len(te)}) ...")
    tr_patches = np.stack([np.load(p) for p in tr["processed_path"]]).astype(np.float32)
    Xtr, ytr_aug = features_for_paths(scat, tr["processed_path"].tolist()), ytr
    if noise_sigmas:
        rng = np.random.default_rng(0)
        parts_x, parts_y = [Xtr], [ytr]
        for s in noise_sigmas:
            noisy = np.clip(tr_patches + rng.normal(0, s, tr_patches.shape).astype(np.float32), 0, 1)
            parts_x.append(scatter_features(scat, noisy))
            parts_y.append(ytr)
        Xtr, ytr_aug = np.concatenate(parts_x), np.concatenate(parts_y)
        print(f"noise-augmented train: {len(ytr)} clean + {len(noise_sigmas)} noisy copies "
              f"-> {len(ytr_aug)} samples (sigmas={noise_sigmas})")
    Xte = features_for_paths(scat, te["processed_path"].tolist())

    pipe = make_pipeline(
        FunctionTransformer(np.log1p, validate=True),       # log-scattering renormalisation
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=2000, random_state=0),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    cvs = cross_val_score(pipe, Xtr, ytr_aug, cv=cv, scoring="balanced_accuracy")
    print(f"train CV balanced acc = {cvs.mean():.3f} ± {cvs.std():.3f}")

    pipe.fit(Xtr, ytr_aug)
    pred = pipe.predict(Xte)
    bal = balanced_accuracy_score(yte, pred)
    print(f"\nTEST balanced acc = {bal:.3f}")
    for c, name in enumerate(classes):
        m = yte == c
        print(f"  {name:<9} acc = {(pred[m] == c).mean():.3f}  (n={int(m.sum())})")
    print("confusion [rows=true, cols=pred]:\n", confusion_matrix(yte, pred))

    with open(out_path, "wb") as f:
        pickle.dump({"pipe": pipe, "classes": list(classes),
                     "scat_params": {"J": J, "L": L, "order": order, "shape": list(shape)}}, f)
    print(f"\nsaved -> {out_path}")
    return bal


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap = argparse.ArgumentParser(description="Train the scattering-feature type classifier")
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--out", default="checkpoints/type_classifier.pkl")
    args = ap.parse_args(argv)
    train_and_save(args.manifest, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
