"""
build_pseudo_labels.py — the ONE command for Milestone 2 (Stage-1 masks).

Reads the manifest from Milestone 1, generates an approximate defect mask for
every image, saves it as a uint8 label map, draws preview overlays, and writes
the mask info back into the manifest.

Run AFTER build_dataset.py:
    py -3.14 -m src.data.build_pseudo_labels --config configs/pseudo_label.yaml

Smoke-test on a handful of images:
    py -3.14 -m src.data.build_pseudo_labels --config configs/pseudo_label.yaml --limit 10
"""

from __future__ import annotations

import os
import sys
import argparse
import random

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .pseudo_label import class_index_map, make_label_map, binary_defect_mask


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_overlay(patch, label_map, out_path, title):
    """patch in grayscale with the pseudo-mask drawn as a colored contour."""
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
    ax[0].imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("amplitude patch")
    ax[1].imshow(patch, cmap="gray", vmin=0, vmax=1)
    if label_map.max() > 0:
        ax[1].contour(label_map > 0, levels=[0.5], colors="red", linewidths=0.8)
    ax[1].set_title(f"pseudo-mask ({int((label_map > 0).sum())} px)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage-1 pseudo-label mask generation")
    ap.add_argument("--config", default="configs/pseudo_label.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap images for a smoke-test")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    manifest_path = cfg["paths"]["manifest"]
    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found. Run build_dataset.py first.")
        return 1
    df = pd.read_csv(manifest_path)

    wanted = set(cfg.get("splits", ["train", "val", "test"]))
    df_use = df[df["split"].isin(wanted)].copy()
    if args.limit is not None:
        df_use = df_use.groupby("class", group_keys=False).head(max(1, args.limit // 2))
    print(f"[1/3] labeling {len(df_use)} images "
          f"(splits={sorted(wanted)})" + (f" [--limit {args.limit}]" if args.limit else ""))

    cls_index = class_index_map(cfg["classes"])
    masks_dir = cfg["paths"]["masks_dir"]
    for c in cfg["classes"]:
        os.makedirs(os.path.join(masks_dir, c), exist_ok=True)

    print("[2/3] generating masks ...")
    info_cols = ["mask_path", "defect_area_px", "defect_area_frac", "n_components",
                 "mask_empty", "used_fallback"]
    results, empties, n_done = [], [], 0
    for _, row in df_use.iterrows():
        patch = np.load(row["processed_path"])
        label_map, info = make_label_map(patch, row["class"], cfg, cls_index)
        out_path = os.path.join(masks_dir, row["class"], row["image_id"] + ".npy")
        np.save(out_path, label_map)

        results.append({"image_id": row["image_id"],
                        "mask_path": out_path.replace("\\", "/"), **info})
        if info["mask_empty"]:
            empties.append(row["image_id"])

        n_done += 1
        if n_done % 100 == 0 or n_done == len(df_use):
            print(f"      {n_done}/{len(df_use)}")

    # merge mask info into the manifest (avoids per-cell dtype coercion)
    res_df = pd.DataFrame(results)
    df = df.drop(columns=[c for c in info_cols if c in df.columns])
    df = df.merge(res_df, on="image_id", how="left")
    df.to_csv(manifest_path, index=False)
    print(f"      manifest updated -> {manifest_path}")

    print("[3/3] saving overlays ...")
    overlays_dir = cfg["paths"]["overlays_dir"]
    os.makedirs(overlays_dir, exist_ok=True)
    n_ov = cfg["overlays"]["n"]
    sample = df_use.groupby("class", group_keys=False).head(max(1, n_ov // len(cfg["classes"])))
    for _, row in sample.head(n_ov).iterrows():
        patch = np.load(row["processed_path"])
        label_map, _ = make_label_map(patch, row["class"], cfg, cls_index)
        save_overlay(patch, label_map,
                     os.path.join(overlays_dir, row["image_id"] + ".png"),
                     f"{row['image_id']}  [{row['class']}, {row['split']}]")
    print(f"      {min(n_ov, len(sample))} overlays -> {overlays_dir}")

    # ── diagnostics summary ──
    labeled = df[df["mask_path"].notna()]
    fr = labeled["defect_area_frac"].astype(float)
    print("\nDONE. Pseudo-label summary:")
    print(f"  images labeled : {len(labeled)}")
    print(f"  empty masks    : {len(empties)}"
          + (f"  (e.g. {empties[:5]})" if empties else "  (good — every image got a mask)"))
    n_fb = int(labeled["used_fallback"].fillna(False).astype(bool).sum())
    print(f"  used fallback  : {n_fb}  (tiny defects rescued from the min-area filter)")
    print(f"  defect-area %  : median={fr.median()*100:.3f}%  "
          f"mean={fr.mean()*100:.3f}%  max={fr.max()*100:.3f}%")
    print(f"  components/img : median={labeled['n_components'].median():.0f}  "
          f"max={int(labeled['n_components'].max())}")
    print(f"  Open {overlays_dir} to eyeball mask quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
