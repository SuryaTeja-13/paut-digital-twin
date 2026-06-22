# Ablation results — two runs kept for comparison

Both runs train the **SCN-Attention U-Net vs a plain Attention U-Net baseline**
(`--no-scattering`) at 100 / 75 / 50 / 25 / 10 % of the training data (80 epochs, Colab T4).
They differ only in augmentation:

| File | Augmentation | Use |
|------|--------------|-----|
| `ablation_results.csv` / `ablation_curve.png` | **base geometric only** (composite/noise OFF) | **Primary** — isolates the scattering prior. This is the run reported in the README and solution-approach. |
| `ablation_results_with_aug.csv` / `ablation_curve_with_aug.png` | production aug **with** composite + noise ON | Archived for comparison only. |

**Why the primary run disables composite/noise:** the data-efficiency ablation must change one
variable (the scattering prior) against data size. The composite (porosity+slag blend) and additive
noise are multi-defect / robustness tricks; at small fractions they consume scarce single-defect
images and destabilise training (the SCN model collapses to Dice ~0.35 at 10% data in the
`*_with_aug` run). Disabling them gives the clean, interpretable comparison. See
`scripts/run_ablation.py` (default = base aug; `--keep-aug` reproduces the with-aug run).

**Type-classifier ablation:** `type_clf_ablation.csv` / `.png` — the scattering-feature MLP trained
on shrinking data (0.88 → 0.72 down to 74 images); the genuine small-data result.
