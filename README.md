# PAUT Explainable-AI Digital Twin

Explainable-AI Digital Twin for weld inspection from Phased Array Ultrasonic Testing (PAUT)
TFM images. The system detects/classifies weld defects, characterizes each defect from its mask,
explains predictions with model-appropriate XAI, and presents everything in a Digital Twin dashboard.

> The full design lives in [`docs/architecture.md`](docs/architecture.md). Project rules and the
> decisions already locked in are in [`design_decisions.md`](design_decisions.md). Read those first.

The core model (built in a later milestone) is an **SCN-Attention U-Net**: a fixed Wavelet
Scattering branch fused with an Attention U-Net.

---

## Setup

```bash
# Windows (this project is developed on Python 3.14)
py -3.14 -m venv .venv
.venv\Scripts\activate
py -3.14 -m pip install -r requirements.txt
```

`requirements.txt` is grouped by milestone — Milestone 1 (preprocessing) only needs
numpy / pandas / pillow / opencv-python / scikit-image / matplotlib / pyyaml.

---

## Build order

This project is built **one module at a time**, in the order of `docs/architecture.md` Part 8.
We pause after each milestone so you can verify it before the next is built.

| # | Milestone | Owner | Status |
|---|-----------|-------|--------|
| 1 | Preprocessing: color→amplitude, crop, normalize, split, manifest | Student 1 | ✅ done |
| 2 | Stage-1 pseudo-label masks | Student 1 | ✅ done |
| 3 | SCN-Attention U-Net (model + training) | Student 2 | ✅ done |
| 4 | Characterization (blobs / PCA / severity) | Student 2 | ✅ done |
| 5 | Explainable AI (Grad-CAM / attention / faithfulness) | Student 3 | ✅ done |
| 6 | Digital Twin + dashboard | Student 4 | ✅ done |

---

## Data note (important)

The dataset is **525 porosity (`G…`) + 525 slag (`GS…`) = 1050 images (balanced)**. Earlier the
porosity folder held 256 `GS…`-named files that were actually slag (byte-identical duplicates),
which made the two classes contradictory and pinned type-classification at 50%; those were removed
and genuine porosity images were added to reach 525 each. `scripts/check_contamination.py` confirms
the folders are clean (no cross-folder duplicates). Defect type (porosity vs slag) is decided by a
scattering-feature classifier (~0.88 balanced accuracy). Class-weighting stays enabled
(`configs/model.yaml` → `train.class_weighting`) as a safeguard.

## Milestone 1 — Preprocessing (done)

Turns every raw TFM image into one normalized single-channel `256×256` float32 amplitude patch,
records a stratified, leakage-safe train/val/test split, and writes a manifest.

### Run it

```bash
# full dataset (1050 images)
py -3.14 -m src.data.build_dataset --config configs/preprocess.yaml

# fast CPU smoke-test on ~10 images
py -3.14 -m src.data.build_dataset --config configs/preprocess.yaml --limit 10
```

### What you get

- `data/processed/<class>/<image_id>.npy` — one normalized amplitude patch per image.
- `data/processed/manifest.csv` — the dataset table (the contract handed to Student 2).
- `data/processed/examples/*.png` — before/after preview figures to eyeball the result.

### Manifest columns

`image_id, class, raw_path, processed_path, source_type, colormap, pixel_to_mm, weld_id,
defect_present, label_type, split, orig_h, orig_w, crop_r0, crop_r1, crop_c0, crop_c1, patch_size`

### Key design choices (and why)

- **Single amplitude channel.** A TFM image is a scalar amplitude field; color is only a colormap.
  Grayscale images take the luminance path; the colormap-inverse path is wired in for *future*
  color data with **no code change** (design_decisions.md decision §1).
- **Border crop.** Raw images are 512×512 with a pure-white matplotlib frame around a ~420×398
  dark data region. We detect and trim the frame automatically (content-based, per image).
- **Per-image min–max normalize** to `[0,1]` — cancels gain/brightness differences and brings out
  faint defects.
- **Group-aware stratified split.** Filenames like `G100_1/_2/_3` are multiple views of the *same*
  weld. The split keeps every view of a weld in **one** split (train *or* val *or* test) so
  near-duplicate views never leak across splits, while still stratifying by class. A built-in guard
  asserts no weld spans two splits.
- **Split happens before any augmentation.** Augmentation (a later module) is applied to the train
  split only — never val/test (design_decisions.md decision §7).
- **Everything is config-driven** (`configs/preprocess.yaml`) — nothing hard-coded, so the same
  code runs locally and in the cloud.

### Tests

```bash
py -3.14 -m tests.test_preprocess         # no pytest needed
# or:  py -3.14 -m pytest tests/ -q
```

---

## Milestone 2 — Stage-1 pseudo-labels (done)

We have no hand-drawn masks yet, so we bootstrap **approximate** defect masks straight from the
amplitude with the classic recipe (architecture.md §4.3): threshold → morphology → fill. These are
a *starting point* for Stage-2 training, not gold ground truth.

### Run it (after Milestone 1)

```bash
py -3.14 -m src.data.build_pseudo_labels --config configs/pseudo_label.yaml
py -3.14 -m src.data.build_pseudo_labels --config configs/pseudo_label.yaml --limit 10   # smoke-test
```

### What you get

- `data/processed/masks/<class>/<image_id>.npy` — a uint8 label map per image
  (`0`=background, `1`=porosity, `2`=slag — design_decisions.md decision §3).
- `data/processed/mask_overlays/*.png` — patch + red mask contour, to eyeball quality.
- The manifest gains `mask_path, defect_area_px, defect_area_frac, n_components, mask_empty,
  used_fallback`.

### Key design choices (and why)

- **Otsu threshold** (with a small floor) adapts per image. The histogram is ~99% background, so
  Otsu lands in the gap and isolates the bright defect core. Triangle/low thresholds grab speckle.
- **OpenCV morphology**, not scipy — `scipy.ndimage.binary_closing` over-erodes thin defects here
  (shrinks slag masks ~3×); `cv2.MORPH_CLOSE` is a correct, extensive closing.
- **Every image gets a mask.** Each image contains a defect (decision §4), so if the standard
  pipeline finds nothing (a tiny scattered defect erased by the min-area filter) we retry keeping
  single pixels. Result on the current data: 0 empty masks, 1 fallback.
- **All splits are labeled.** With no hand labels yet, val/test masks are approximate too — replace
  with a small hand-labeled set later for an honest segmentation score.

> Single-class-per-image: each image is porosity OR slag, so every defect pixel in it gets that one
> class index. Mixed-defect images would need per-instance class labels (a later extension).

### Tests

```bash
py -3.14 -m tests.test_pseudo_label
```

---

## Milestone 3 — SCN-Attention U-Net (done)

The core multi-task model (architecture.md Part 5): a fixed **Wavelet Scattering** branch fused
with an **Attention U-Net**, with three heads.

```
input 1x256x256
   ├─ Attention U-Net encoder ─ E1..E4, bottleneck ─┐
   └─ Wavelet Scattering (fixed) ─ S + global vec ──┤  per-scale fusion:
                                                     │  concat(E_i, resize(1x1Conv(S)))
                                                     │  -> 1x1Conv -> CBAM
   attention-gated decoder  <───────────────────────┘
   ├─ Head 1  segmentation  (bg / porosity / slag)   3x256x256
   ├─ Head 2  classification (porosity vs slag)
   └─ Head 3  size regression (optional, off)
```

Files: [src/models/blocks.py](src/models/blocks.py) (ConvBlock, CBAM, AttentionGate, UpBlock),
[scattering.py](src/models/scattering.py), [scn_attention_unet.py](src/models/scn_attention_unet.py),
[losses.py](src/models/losses.py), [metrics.py](src/models/metrics.py),
[train.py](src/models/train.py), and [src/data/dataset.py](src/data/dataset.py).

### Train

```bash
# CPU smoke-test (tiny subset, 2 epochs) — just proves the pipeline runs
py -3.14 -m src.models.train --config configs/model.yaml --limit 12 --epochs 2

# prove the model LEARNS: overfit a few images, watch train dice_fg climb
py -3.14 -m src.models.train --config configs/model.yaml --overfit 6 --epochs 40

# full run — do this on a Colab/Kaggle GPU (see notebooks/train_colab.ipynb)
py -3.14 -m src.models.train --config configs/model.yaml

# plain Attention-U-Net baseline (no scattering) for the data-ablation comparison
py -3.14 -m src.models.train --config configs/model.yaml --no-scattering
```

Best checkpoint (by val foreground-Dice) is saved to `checkpoints/`. Loss is
`0.6*DiceFocal + 0.3*CE + 0.1*boundary` (decision §10) — all weights in `configs/model.yaml`.

### Key design choices (and why)

- **Scattering filters are frozen** (decision §9): fixed filters can't overfit tiny data — that is
  the whole reason for this architecture. Only the 1x1 projection convs in the fusion are learnable.
- **GroupNorm, not BatchNorm**: batch sizes are tiny on limited data, where BatchNorm stats are noisy.
- **CBAM + Attention Gates**: suppress busy weld background; their attention maps are reused later as
  free, intrinsic explanations for Student 3.
- **kymatio import**: we import the 2D torch frontend directly, because `kymatio.torch` eagerly
  imports a 3D module that needs `scipy.special.sph_harm` (removed in scipy ≥ 1.16).
- **`use_scattering: false`** turns the model into a plain Attention-U-Net for the ablation that
  proves the scattering prior helps in the low-data regime (architecture.md §5.11).

### Tests

```bash
py -3.14 -m tests.test_model
```

---

## Milestone 4 — Characterization (done)

Measures every defect from its mask (architecture.md §5.7, decision §11): size, shape,
orientation (PCA), location, and a severity score — one record per defect instance.

### Run

```bash
# uses the trained model's predicted masks
py -3.14 -m src.characterize.run --config configs/characterize.yaml

# or the Stage-1 pseudo-masks (no checkpoint needed)
py -3.14 -m src.characterize.run --config configs/characterize.yaml --mask-source pseudo
```

Outputs per image a `data/processed/characterization/<id>.json` (the **S2 → S4 contract**:
type, length/width/area, orientation, aspect ratio, eccentricity, centroid, confidence,
severity), a `_summary.csv`, and annotated overlays in `data/processed/char_overlays/`.

### Key points

- **Defect type comes from the scattering classifier** (~0.88 balanced), applied at image level;
  the segmentation supplies *detection* (where the defect is). The two in-network neural routes —
  seg-derived type (~0.72) and the global-pooled classification head (~0.72) — are both weaker
  than the scattering classifier, so they are kept and reported only, with "improve its pooling"
  logged as a future tuning
  item. This refines decision §11.
- **Multi-defect**: connected components → one characterization record each.
- **pixel_to_mm defaults to 1.0** → values are in pixels; set the real scale later (no code change).
  Scale-free fields (orientation, aspect ratio, eccentricity) are already correct.
- **Severity is a tunable placeholder** (`configs/characterize.yaml`), NOT an acceptance rule —
  calibrate to ISO 5817 / ASME with a domain expert before any pass/fail use (decision §11).

### Tests
```bash
py -3.14 -m tests.test_characterize
```

---

## Milestone 5 — Explainable AI (done)

Model-appropriate XAI (architecture.md Part 6), matched to each head:

- **Seg-Grad-CAM / Grad-CAM++** on the last decoder conv for the **segmentation** head.
- **Grad-CAM** on the bottleneck for the **classification** head.
- **Intrinsic attention** — the model's own CBAM spatial maps and Attention-Gate
  coefficients, surfaced for free (faithful by construction).
- **Faithfulness metrics** — deletion / insertion AUC + pointing-game → a single
  **trust score** per explanation (the "standards" requirement).

```bash
py -3.14 -m src.xai.run --config configs/xai.yaml --split test
```
Outputs a 4-panel overlay (amplitude · Grad-CAM · attention · GT mask) and a per-image
JSON with the trust score, in `data/processed/xai/` and `data/processed/xai_overlays/`.

On a 16-image validation sample: mean **trust 0.91**, deletion AUC 0.12 (lower=better), insertion AUC
0.86 (higher=better), pointing-game 1.00 — the Grad-CAM explanations are faithful. (Reproduce:
`py -3.14 -m src.xai.run --config configs/xai.yaml --split val --limit 16`; writes
`data/processed/xai/_summary.csv`.)

> SHAP on the classification head (architecture.md §6.1) is left as an optional add — the
> classification head is the weak one we route around (type comes from the scattering classifier), and SHAP adds a
> heavy dependency. The gradient methods above are the doc's *primary* choice for the segmentation
> head, which is the strong one.

### Tests
```bash
py -3.14 -m tests.test_xai
```

---

## Defect type — scattering classifier

The U-Net detects defects well but types them poorly (porosity ↔ slag confusion; porosity type was
~0.50 = chance). Type is a whole-image texture property, so a **small classifier on the global
Wavelet-Scattering vector** decides porosity-vs-slag instead. Using **log-scattering** (the standard
Mallat/Bruna log-renormalisation) and a **tiny one-hidden-layer MLP** — i.e. the "scattering + small
neural classifier" of the IWSCN reference paper — it reaches **test balanced 0.88 (porosity 0.86,
slag 0.90)**, up from 0.80 for a plain logistic regression. Selection is by 5-fold CV on the train
split; the test split is scored once (CV 0.87 ≈ test 0.88, so no overfitting). The scattering filters
stay fixed (J=2, L=8, order=2) — **only the small classifier changed, the U-Net is untouched**. The
pipeline uses **segmentation for detection** + **scattering classifier for type**.

```bash
py -3.14 -m src.models.type_classifier --out checkpoints/type_classifier.pkl   # trains on CPU in minutes
```

## Model evaluation

```bash
py -3.14 -m src.models.evaluate --ckpt checkpoints/scn_attn_unet_best.pt --split test
```
Reports per-class Dice/IoU, classification accuracy + balanced accuracy + confusion matrix,
and the near-empty-mask ("no defect") rate.

Current checkpoint (test split, 158 images; composite + noise augmentation + Student-1 steel-block
defects in the train split): foreground Dice **0.69** (porosity 0.66, slag 0.71 — both classes up);
seg-derived type balanced accuracy **0.72**. The independent **scattering classifier** (0.88) is
still used for type because it is the most reliable; the neural classification head also recovered
to **~0.72 balanced** in this run and is reported alongside it.

---

## Milestone 6 — Digital Twin + dashboard (done)

The integration layer and the visible end-product (architecture.md Part 7).

**One-click pipeline** — [src/pipeline.py](src/pipeline.py): `image → preprocess → model →
characterize → XAI → health`, one call returns everything.
```bash
py -3.14 -m src.pipeline --image data/raw/slag/GS1.jpg --out defects.json
```

**Batch / mini-batch inference** — [scripts/batch_infer.py](scripts/batch_infer.py): run a whole
folder through the pipeline with one model forward per mini-batch (`Pipeline.analyze_batch`,
`predict_batch`). Writes a per-image table (type, defect count, health, PASS/REVIEW/FAIL).
```bash
py -3.14 -m scripts.batch_infer --dir data/raw/slag --limit 15 --batch-size 8 --time-compare
```

**Digital-twin health** — [src/twin/twin.py](src/twin/twin.py): rolls per-defect severities into a
health index (1 − Σseverity/capacity) and a PASS / REVIEW / FAIL status. *(Thresholds are
placeholders — calibrate to ISO 5817 / ASME with an expert; design_decisions.md §3.13.)*

**Dashboard** — [src/dashboard/app.py](src/dashboard/app.py):
```bash
py -3.14 -m streamlit run src/dashboard/app.py
```
Upload a TFM image (color or grayscale) → weld map with severity-colored defect markers, a
Seg-Grad-CAM heatmap + trust score, a per-defect table and cards (size/orientation/severity/
confidence), and the weld-health panel. Verified: pipeline runs end-to-end (GS1 → 2 defects,
health 0.84, PASS, XAI trust 0.95) and the app boots and serves.

**Continuous feed (real-time inspection)** — the dashboard's *Continuous feed* mode auto-processes
a whole folder of welds in sequence as a live "video feed" (no manual one-at-a-time selection),
showing each frame's detection, the digital-twin health with an explicit **calculation proof**
(`health = 1 − min(1, Σseverity/capacity)`), and a running health timeline + PASS/REVIEW/FAIL
tally. Build the 15-frame demo set (12 defect + 3 no-defect) first:
```bash
py -3.14 -m scripts.make_demo_feed     # -> data/processed/demo_feed/
```

### Tests
```bash
py -3.14 -m tests.test_twin
```

---

## Data-Ablation Results

SCN-Attention U-Net vs plain Attention U-Net baseline across data fractions (60 epochs, Colab T4).

| Data used | SCN val Dice | Plain U-Net val Dice |
|-----------|-------------|----------------------|
| 10% (74 imgs) | **0.541** | 0.514 |
| 25% (184 imgs) | 0.540 | 0.567 |
| 50% (368 imgs) | 0.580 | 0.664 |
| 100% (735 imgs) | 0.642 | **0.689** |

**Key finding:** At 10% of training data the SCN model outperforms the plain baseline (0.541 vs
0.514), showing the fixed wavelet-scattering prior acts as a regulariser in the low-data regime
that is typical of PAUT inspection. As data increases, the plain U-Net surpasses SCN — the added
fusion complexity becomes a burden once sufficient data is available. In these short ablation runs
(60 epochs each) both models' classification heads stayed near chance (~50% val accuracy); the
fully-trained production head is also weak (~0.50–0.65 depending on the run) — far below the
independent scattering-feature classifier (0.88 balanced), which is why type is decided by it.

Curve: [`data/processed/ablation/ablation_curve.png`](data/processed/ablation/ablation_curve.png)
Raw numbers: [`data/processed/ablation/ablation_results.csv`](data/processed/ablation/ablation_results.csv)

### Type-classifier data-ablation (the scattering prior in the low-data regime)

The same §5.11 curriculum applied to the **type classifier** (log-scattering + tiny MLP), trained on
shrinking fractions of the train split, scored on the **full** test split (5 random subsamples per
fraction, mean ± std). It runs on CPU in minutes because the scattering features are fixed.

| Training data | Balanced acc | porosity | slag |
|---------------|-------------|----------|------|
| 100% (735) | 0.880 | 0.863 | 0.897 |
| 50% (368) | 0.828 | 0.785 | 0.872 |
| 25% (184) | 0.786 | 0.755 | 0.818 |
| 10% (74) | **0.721** | 0.665 | 0.777 |

**Finding:** accuracy degrades *gracefully* and stays **well above chance (0.50)** even at 10% of the
data (74 images → 0.72). This is the small-data robustness the fixed wavelet-scattering prior is
chosen for. Reproduce: `py -3.14 -m scripts.ablation_type_classifier`.

Curve: [`data/processed/ablation/type_clf_ablation.png`](data/processed/ablation/type_clf_ablation.png)

---

## Run on a different machine / the cloud

All paths and hyperparameters are in `configs/`. To train later on Colab/Kaggle GPU, swap the
config — the code detects the device and never hard-codes `.cuda()` (design_decisions.md §5).
