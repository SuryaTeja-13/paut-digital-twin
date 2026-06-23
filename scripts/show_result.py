"""show_result.py — pop up an interactive image window for one PAUT scan.

Runs the model (Student 2) and, with --xai, the explainability (Student 3) on a single
image and opens a matplotlib window so the result can be shown live during evaluation —
no saved files, no dashboard. Uses an interactive backend (a window pops up).

Examples:
    # Student 2 — model only (input + segmentation + weld map)
    py -3.14 scripts/show_result.py --image data/raw/slag/GS1.jpg --no-xai

    # Student 3 — add Grad-CAM + the model's attention
    py -3.14 scripts/show_result.py --image data/raw/porosity/G12.jpg
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOTE: do NOT force the Agg backend here — we want a real pop-up window.
import matplotlib.pyplot as plt

from src.pipeline import Pipeline
from src.twin.twin import SEVERITY_COLOR


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pop up an image window of the model/XAI result")
    ap.add_argument("--image", required=True, help="path to a PAUT image")
    ap.add_argument("--checkpoint", default="checkpoints/scn_attn_unet_best.pt")
    ap.add_argument("--no-xai", action="store_true", help="model only (skip Grad-CAM/attention)")
    a = ap.parse_args(argv)

    run_xai = not a.no_xai
    pipe = Pipeline(checkpoint=a.checkpoint)
    res = pipe.analyze(a.image, run_xai=run_xai)

    patch, seg, defects = res["patch"], res["seg"], res["defects"]
    hh = res["health"]

    panels = ["input", "segmentation"]
    if run_xai and "xai" in res:
        panels += ["gradcam", "attention"]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]

    for ax, kind in zip(axes, panels):
        ax.axis("off")
        if kind == "input":
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax.set_title("Input PAUT image")
        elif kind == "segmentation":
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax.contour(seg > 0, levels=[0.5], colors="cyan", linewidths=0.8)
            for d in defects:
                x, y = d["centroid_px"]
                ax.scatter([x], [y], s=130, edgecolors="white",
                           c=SEVERITY_COLOR.get(d["severity"], "#fff"), zorder=3)
                ax.text(x + 5, y, str(d["id"]), color="white", fontsize=9, weight="bold")
            ax.set_title(f"Segmentation — {res['defect_type'] or 'no defect'} ({len(defects)} defects)")
        elif kind == "gradcam":
            x = res["xai"]
            ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
            ax.imshow(x["cam"], cmap="jet", alpha=0.5)
            ax.set_title(f"Grad-CAM (trust {x['trust_score']:.2f})")
        elif kind == "attention":
            att = res["xai"].get("attention", {})
            if att:
                key = "gate_up1" if "gate_up1" in att else list(att)[0]
                ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
                ax.imshow(att[key], cmap="jet", alpha=0.5)
                ax.set_title(f"Model attention ({key})")
            else:
                ax.set_title("attention n/a")

    fig.suptitle(f"{os.path.basename(a.image)}   |   health {hh['health_index']:.3f}  "
                 f"·  {hh['status']}", fontsize=13, weight="bold")
    fig.tight_layout()
    print(f"image: {res['image']} | type: {res['defect_type']} | defects: {hh['n_defects']} | "
          f"health: {hh['health_index']:.3f} ({hh['status']})")
    print("Close the image window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
