"""
import_external.py — convert Student-1's steel-block artificial-defect images to our format.

These are jet-colormap TFM images of ARTIFICIAL porosity (drilled holes) and slag (notches)
in a steel reference block. They differ from our grayscale weld dataset in three ways that this
importer fixes so they are compatible:
  1. colour (jet)        -> recover scalar amplitude (colormap_inverse, the path already in
                            preprocess.py for exactly this case)
  2. backwall echo       -> the bright horizontal line at the bottom is the block's back surface,
                            NOT a defect; we crop the bottom band (and the top array fringe) so the
                            model never learns to flag it
  3. arbitrary size      -> resize to 256x256 and min-max normalise, matching the dataset

Outputs converted grayscale patches to data/processed/external/<class>/ (both .npy for the model
and .png to view), then runs the CURRENT model on them as an independent check.

Run:  py -3.14 -m scripts.import_external
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.transform import resize

from src.data.preprocess import colormap_inverse
from src.models.infer import load_model, predict_patch

SRC = r"C:\Users\Surya Teja\Downloads\raw images"
OUT = "data/processed/external"
CKPT = "checkpoints/scn_attn_unet_best.pt"
TOP_CROP = 0.08          # trim top array-fringe artefact
BOT_CROP = 0.20          # trim bottom backwall echo

# my labels, confirmed by the user (artificial defects on a steel block)
LABELS = {
    "Figure_3.png": "porosity",
    "Figure_9.png": "porosity",
    "Figure_1 (1).png": "porosity",
    "Figure_1.png": "porosity",
    "Figure_2.png": "porosity",
    "Figure_5.png": "slag",
    "Figure_5 (1).png": "slag",
    # "Figure_3 (1).png" excluded — too faint/noisy
}


def convert(path):
    rgb = mpimg.imread(path)                       # HxWx{3,4} float 0..1 or uint8
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8)
    rgb = rgb[:, :, :3]
    amp = colormap_inverse(rgb, "jet")             # -> 0..255 scalar amplitude
    h = amp.shape[0]
    amp = amp[int(TOP_CROP * h):int((1 - BOT_CROP) * h), :]   # drop fringe + backwall
    patch = resize(amp, (256, 256), preserve_range=True).astype(np.float32)
    mn, mx = patch.min(), patch.max()
    patch = (patch - mn) / (mx - mn + 1e-8)        # min-max to [0,1] like the dataset
    return patch.astype(np.float32)


def main():
    model, _ = load_model(CKPT)
    for c in ("porosity", "slag"):
        os.makedirs(os.path.join(OUT, c), exist_ok=True)
    gallery, rows = [], []
    for fname, cls in LABELS.items():
        src = os.path.join(SRC, fname)
        if not os.path.exists(src):
            print(f"  MISSING: {src}")
            continue
        patch = convert(src)
        stem = fname.replace(".png", "").replace(" ", "_").replace("(", "").replace(")", "")
        np.save(os.path.join(OUT, cls, f"EXT_{stem}.npy"), patch)
        mpimg.imsave(os.path.join(OUT, cls, f"EXT_{stem}.png"), patch, cmap="gray", vmin=0, vmax=1)

        pred = predict_patch(model, patch)
        seg = pred["seg"]
        por_px, slag_px = int((seg == 1).sum()), int((seg == 2).sum())
        detected = (por_px + slag_px) >= 10
        rows.append({"file": fname, "label": cls, "detected": detected,
                     "porosity_px": por_px, "slag_px": slag_px})
        print(f"  {fname:<18} [{cls:<8}] detected={detected} (por {por_px}px, slag {slag_px}px)")
        gallery.append((patch, seg, f"{fname}\n[{cls}]"))

    # gallery: converted patch (top) + model prediction (bottom)
    n = len(gallery)
    fig, ax = plt.subplots(2, n, figsize=(2.5 * n, 5.4))
    for j, (patch, seg, title) in enumerate(gallery):
        ax[0, j].imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(title, fontsize=7)
        ax[1, j].imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax[1, j].imshow(np.ma.masked_where(seg == 0, seg), cmap="autumn", alpha=0.7, vmin=1, vmax=2)
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle("External steel-block defects: converted input (top) + model detection (bottom)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "external_gallery.png"), dpi=110)
    plt.close(fig)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "external_results.csv"), index=False)
    print(f"\nconverted {len(rows)} images -> {OUT}/<class>/ "
          f"(porosity {sum(r['label']=='porosity' for r in rows)}, "
          f"slag {sum(r['label']=='slag' for r in rows)})")
    print(f"model detected a defect in {df['detected'].mean()*100:.0f}% of them")
    print(f"gallery -> {OUT}/external_gallery.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
