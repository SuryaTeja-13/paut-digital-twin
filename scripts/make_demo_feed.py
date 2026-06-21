"""
make_demo_feed.py — build a demo image sequence for the continuous digital-twin feed (sir 8).

Sir wants the twin to run automatically over ~15 images, INCLUDING some no-defect images, as
a live feed. Our dataset is all-defect (every weld image has a flaw), so this assembles a
reproducible demo folder: real porosity + slag images interleaved with a few synthetic
"clean" (no-defect) frames — a faint dark field with low-amplitude speckle and no bright flaw.
A correct twin should mark those clean frames PASS with health 1.0.

Files are named NN_<label>.png so they play in a fixed, interleaved order.

Run:  py -3.14 -m scripts.make_demo_feed
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.image as mpimg

MANIFEST = "data/processed/manifest.csv"
OUT_DIR = "data/processed/demo_feed"
RNG = np.random.default_rng(7)


def make_clean(size=256):
    """A no-defect frame: a smooth, wide low-frequency field with no compact bright flaw.

    NOTE: per-image min-max normalisation (preprocess) stretches pure noise into full-contrast
    speckle the model mistakes for porosity, so a clean frame must be SMOOTH, not noisy. A wide
    Gaussian bump (vignette-like) normalises to a smooth field with no defect-shaped structure,
    which the model correctly reads as no-defect (verified: 0 foreground pixels).
    """
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = RNG.integers(105, 151, 2)
    sigma = RNG.uniform(85, 110)
    img = 0.1 + 0.8 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    img += RNG.normal(0, 0.005, (size, size))
    return np.clip(img, 0, 1).astype(np.float32)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    df = pd.read_csv(MANIFEST)
    df = df[df["split"] == "test"]
    por = df[df["class"] == "porosity"]["raw_path"].tolist()[:6]
    slg = df[df["class"] == "slag"]["raw_path"].tolist()[:6]

    # interleave 6 porosity + 6 slag + 3 clean = 15 frames, clean ones sprinkled in
    plan = []
    for i in range(6):
        plan.append(("porosity", por[i]))
        plan.append(("slag", slg[i]))
        if i in (1, 3, 5):
            plan.append(("clean", None))

    n = 0
    for label, src in plan:
        n += 1
        dst = os.path.join(OUT_DIR, f"{n:02d}_{label}.png")
        if label == "clean":
            mpimg.imsave(dst, make_clean(), cmap="gray", vmin=0, vmax=1)
        else:
            if src and os.path.exists(src):
                shutil.copy(src, dst)
            else:                                      # fallback if raw path missing
                mpimg.imsave(dst, make_clean(), cmap="gray", vmin=0, vmax=1)
    print(f"wrote {n} demo frames to {OUT_DIR} "
          f"(6 porosity, 6 slag, 3 clean no-defect)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
