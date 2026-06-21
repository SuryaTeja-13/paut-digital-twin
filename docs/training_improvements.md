# Training improvements (augmentation) — wired in, retrain on GPU to activate

These changes target the weak points sir flagged (fusion, noise robustness) by enriching the
**training data**, not the model. The SCN-Attention U-Net architecture and the fixed scattering
filters are **unchanged** — so there is no risk of the collapse a redesign would cause. The
improvements only take effect after a **retrain on GPU** (Colab/Kaggle); the current production
checkpoint is untouched until then.

## What was added (`configs/model.yaml → augment`, train split only)

| Knob | Value | Fixes | How it helps |
|------|-------|-------|--------------|
| `noise_p` / `noise_sigma` | 0.3 / 0.04 | Noise robustness (#5) | Adds additive Gaussian noise to training images so the seg model learns to keep finding defects under acquisition noise. |
| `composite_p` | 0.25 | Multi-defect fusion (#10) | With 25% probability, a training image is max-blended with one of the **other** class and their masks are unioned — so the model sees **both porosity and slag in one image** and learns to segment both. Every original image is single-type, which is exactly why fusion detection was weak. |

Both are train-only (never val/test) and gated by config (default off → no behaviour change unless
enabled). Verified: with `composite_p=1.0` every sampled training mask contains both flaw classes.

## What this does NOT fix
- **Porosity Dice (~0.45):** the root cause is pseudo-labels (they shrink porosity to tiny cores).
  The real lever is hand-labelling ~30–50 images. Composite/noise augmentation may help a little but
  is not the main fix.
- **Extreme-noise type accuracy:** a fundamental signal-to-noise limit (see the type-classifier
  trade-off in the robustness commit) — not addressed by training augmentation.

## How to activate (one Colab/Kaggle GPU run)

```bash
# in the Colab notebook (notebooks/train_colab.ipynb), same as before — the new augmentation
# is read from configs/model.yaml automatically:
py -3.14 -m src.models.train --config configs/model.yaml
# then re-evaluate and refresh the experiment numbers:
py -3.14 -m src.models.evaluate --ckpt checkpoints/scn_attn_unet_best.pt --split test
py -3.14 -m scripts.multidefect_fusion --pairs 8     # expect both-detected to rise
py -3.14 -m scripts.robustness --limit 10            # expect the noise curve to fall less steeply
```

After the run, commit the new checkpoint + refreshed `model_metrics.json` and the experiment
outputs. The numbers will then reflect the improved, incubated model.
