"""
build_dataset.py — the ONE command for Milestone 1 (Student 1).

What it does:
  1. Scans data/raw/<class>/ for images.
  2. Computes a stratified, group-aware train/val/test split  (BEFORE any
     augmentation — no leakage).
  3. Preprocesses every image to a normalized 1-channel 256x256 float32 patch
     and saves it as .npy under data/processed/<class>/.
  4. Writes data/processed/manifest.csv (the contract handed to Student 2).
  5. Saves a few before/after preview PNGs so you can eyeball the result.

Run (full dataset):
    py -3.14 -m src.data.build_dataset --config configs/preprocess.yaml

Quick CPU smoke-test on ~10 images:
    py -3.14 -m src.data.build_dataset --config configs/preprocess.yaml --limit 10

Everything is driven by the YAML config; nothing is hard-coded (design_decisions.md §5).
"""

from __future__ import annotations

import os
import sys
import glob
import json
import math
import argparse
import random

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")               # headless: just write files, no GUI window
import matplotlib.pyplot as plt

from .preprocess import load_rgb, detect_source_type, to_amplitude, process_image
from .split import assign_splits, split_summary, weld_id_from_stem


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def scan_raw(cfg: dict, limit: int | None) -> list[dict]:
    """
    Build the list of records from data/raw/<class>/. image_id and weld_id are
    prefixed with the class so the two folders can never collide.

    --limit N caps the TOTAL images, split as evenly as possible across classes,
    for a fast CPU smoke-test.
    """
    raw_dir = cfg["paths"]["raw_dir"]
    classes = cfg["classes"]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

    per_class_cap = None
    if limit is not None:
        per_class_cap = max(1, math.ceil(limit / len(classes)))

    records = []
    for cls in classes:
        files = []
        for e in exts:
            files += glob.glob(os.path.join(raw_dir, cls, e))
            files += glob.glob(os.path.join(raw_dir, cls, e.upper()))
        files = sorted(set(files))
        if per_class_cap is not None:
            files = files[:per_class_cap]
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            records.append({
                "image_id": f"{cls}__{stem}",
                "class": cls,
                "raw_path": f.replace("\\", "/"),
                "weld_id": f"{cls}__{weld_id_from_stem(stem)}",
            })
    return records


def verify_no_group_leakage(records, split_map, group_by_weld: bool) -> None:
    """Assert that no weld group spans more than one split (the leakage guard)."""
    if not group_by_weld:
        return
    weld_splits = {}
    for r in records:
        weld_splits.setdefault(r["weld_id"], set()).add(split_map[r["image_id"]])
    leaked = {w: s for w, s in weld_splits.items() if len(s) > 1}
    if leaked:
        raise RuntimeError(
            f"LEAKAGE: {len(leaked)} weld(s) span multiple splits, e.g. "
            f"{list(leaked.items())[:3]}")


def save_examples(example_records, cfg: dict) -> None:
    """Save before/after preview figures: raw RGB | amplitude+crop | final patch."""
    out_dir = cfg["paths"]["examples_dir"]
    os.makedirs(out_dir, exist_ok=True)
    for r in example_records:
        rgb = load_rgb(r["raw_path"])
        st = detect_source_type(rgb, cfg["amplitude"].get("gray_tolerance", 6))
        amp = to_amplitude(rgb, cfg["amplitude"], st)
        patch = np.load(r["processed_path"])

        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        ax[0].imshow(rgb); ax[0].set_title(f"raw ({st})\n{r['orig_h']}x{r['orig_w']}")
        ax[1].imshow(amp, cmap="gray"); ax[1].set_title("amplitude (pre-crop)")
        ax[2].imshow(patch, cmap="gray", vmin=0, vmax=1)
        ax[2].set_title(f"final patch {patch.shape[0]}x{patch.shape[1]}\n[{r['class']}, {r['split']}]")
        for a in ax:
            a.axis("off")
        fig.suptitle(r["image_id"], fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{r['image_id']}.png"), dpi=90)
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PAUT preprocessing -> patches + manifest")
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total images (split evenly across classes) for a CPU smoke-test")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seeds(cfg["seed"])

    print(f"[1/5] scanning {cfg['paths']['raw_dir']} ...")
    records = scan_raw(cfg, args.limit)
    if not records:
        print("ERROR: no images found. Check data/raw/<class>/ paths in the config.")
        return 1
    n_welds = len({r["weld_id"] for r in records})
    print(f"      found {len(records)} images across {len(cfg['classes'])} classes, {n_welds} weld groups")
    if args.limit is not None:
        print(f"      (--limit {args.limit} -> smoke-test subset)")

    print("[2/5] computing split (stratified by class, group-aware by weld) ...")
    sp = cfg["split"]
    split_map = assign_splits(
        records, sp["ratios"], cfg["seed"],
        stratify_by_class=sp.get("stratify_by_class", True),
        group_by_weld=sp.get("group_by_weld", True),
    )
    verify_no_group_leakage(records, split_map, sp.get("group_by_weld", True))
    print(split_summary(records, split_map))

    print("[3/5] preprocessing images -> .npy patches ...")
    processed_dir = cfg["paths"]["processed_dir"]
    for cls in cfg["classes"]:
        os.makedirs(os.path.join(processed_dir, cls), exist_ok=True)

    rows, n_done = [], 0
    for r in records:
        patch, meta = process_image(r["raw_path"], cfg)
        out_path = os.path.join(processed_dir, r["class"], r["image_id"] + ".npy")
        np.save(out_path, patch)
        rows.append({
            "image_id": r["image_id"],
            "class": r["class"],
            "raw_path": r["raw_path"],
            "processed_path": out_path.replace("\\", "/"),
            "source_type": meta["source_type"],
            "colormap": meta["colormap"],
            "pixel_to_mm": cfg["pixel_to_mm"],
            "weld_id": r["weld_id"],
            "defect_present": True,          # every image here contains a defect (design_decisions.md §4 dec.4)
            "label_type": cfg["label_type"], # masks not generated yet
            "split": split_map[r["image_id"]],
            "orig_h": meta["orig_h"], "orig_w": meta["orig_w"],
            "crop_r0": meta["crop_r0"], "crop_r1": meta["crop_r1"],
            "crop_c0": meta["crop_c0"], "crop_c1": meta["crop_c1"],
            "patch_size": cfg["patch_size"],
        })
        n_done += 1
        if n_done % 100 == 0 or n_done == len(records):
            print(f"      {n_done}/{len(records)}")

    print("[4/5] writing manifest ...")
    manifest_path = cfg["paths"]["manifest"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"      manifest -> {manifest_path}  ({len(df)} rows)")

    print("[5/5] saving before/after preview figures ...")
    # pick a spread of examples: a few of each class, mixing splits
    n_ex = cfg["examples"]["n"]
    ex = []
    for cls in cfg["classes"]:
        cls_rows = [row for row in rows if row["class"] == cls]
        ex += cls_rows[: max(1, n_ex // len(cfg["classes"]))]
    save_examples(ex[:n_ex], cfg)
    print(f"      {min(n_ex, len(ex))} previews -> {cfg['paths']['examples_dir']}")

    # quick sanity readout on the first patch
    first = np.load(rows[0]["processed_path"])
    print("\nDONE. Sanity check on first patch:")
    print(f"  shape={first.shape}  dtype={first.dtype}  "
          f"min={first.min():.3f}  max={first.max():.3f}  mean={first.mean():.3f}")
    print(f"  Open {cfg['paths']['examples_dir']} to see before/after images.")
    print(f"  Open {manifest_path} to see the dataset table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
