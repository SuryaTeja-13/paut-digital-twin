"""
run.py — characterize every image and emit the S2 -> S4 hand-off (defects.json).

For each image it gets a segmentation label map (from the trained model, or the
Stage-1 pseudo-mask), measures each defect, writes a per-image defects.json, an
annotated overlay, and a combined summary CSV for the dashboard.

Run (model-predicted masks):
    py -3.14 -m src.characterize.run --config configs/characterize.yaml

Use the Stage-1 pseudo-masks instead (no checkpoint needed):
    py -3.14 -m src.characterize.run --config configs/characterize.yaml --mask-source pseudo

Smoke-test on a few images:
    py -3.14 -m src.characterize.run --config configs/characterize.yaml --limit 8
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .characterize import characterize_mask, summarize


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_overlay(patch, defects, out_path, title):
    """Grayscale patch + an ellipse and label per defect (color by severity)."""
    color = {"minor": "lime", "moderate": "yellow", "critical": "red"}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    for d in defects:
        cx, cy = d["centroid_px"]
        col = color.get(d["severity"], "cyan")
        e = Ellipse((cx, cy), width=max(d["width_mm"], 3), height=max(d["length_mm"], 3),
                    angle=-d["orientation_deg"], fill=False, edgecolor=col, lw=1.3)
        ax.add_patch(e)
        ax.text(cx, cy - 6, f"{d['id']}:{d['type'][:3]}", color=col, fontsize=7, ha="center")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=95)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Characterize defects from masks")
    ap.add_argument("--config", default="configs/characterize.yaml")
    ap.add_argument("--mask-source", choices=["model", "pseudo"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="restrict to one split (train/val/test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mask_source = args.mask_source or cfg.get("mask_source", "model")
    classes = cfg["classes"]
    s = cfg["pixel_to_mm"]

    df = pd.read_csv(cfg["paths"]["manifest"])
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.groupby("class", group_keys=False).head(max(1, args.limit // len(classes)))
    df = df.reset_index(drop=True)

    model = None
    if mask_source == "model":
        from ..models.infer import load_model, predict_patch
        print(f"loading model: {cfg['checkpoint']}")
        model, _ = load_model(cfg["checkpoint"])
        predict = predict_patch
    print(f"characterizing {len(df)} images  (mask_source={mask_source})")

    out_dir = cfg["paths"]["out_dir"]
    ov_dir = cfg["paths"]["overlays_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)

    rows, n_ov = [], cfg["overlays"]["n"]
    for idx, row in df.iterrows():
        patch = np.load(row["processed_path"])
        seg_prob = None
        if mask_source == "model":
            r = predict(model, patch)
            label_map = r["seg"]
            seg_prob = r["seg_prob"]
        else:
            label_map = np.load(row["mask_path"]).astype(np.int64)

        defects = characterize_mask(label_map, classes, pixel_to_mm=s,
                                    severity_cfg=cfg["severity"],
                                    min_area_px=cfg["min_area_px"], seg_prob=seg_prob)
        summ = summarize(defects, s)

        record = {"image_id": row["image_id"], "true_class": row["class"],
                  "split": row["split"], "mask_source": mask_source,
                  "pixel_to_mm": s, "summary": summ, "defects": defects}
        with open(os.path.join(out_dir, row["image_id"] + ".json"), "w") as f:
            json.dump(record, f, indent=2)

        rows.append({"image_id": row["image_id"], "true_class": row["class"],
                     "split": row["split"], **summ})
        if idx < n_ov:
            save_overlay(patch, defects, os.path.join(ov_dir, row["image_id"] + ".png"),
                         f"{row['image_id']}  {summ['dominant_type']} "
                         f"({summ['n_defects']} defects, {summ['max_severity']})")

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # quick readout: how often the seg-derived dominant type matches the true label
    matched = (summary_df["dominant_type"] == summary_df["true_class"]).mean()
    print(f"\nDONE. per-image defects.json -> {out_dir}")
    print(f"  summary csv -> {summary_csv}  ({len(summary_df)} rows)")
    print(f"  overlays -> {ov_dir}")
    print(f"  median defects/image: {int(summary_df['n_defects'].median())}  "
          f"max: {int(summary_df['n_defects'].max())}")
    print("  severity mix: " +
          ", ".join(f"{k}={v}" for k, v in summary_df['max_severity'].value_counts().items()))
    print(f"  seg-derived dominant type matches true label: {matched:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
