"""
run.py — produce explanations + faithfulness/trust scores (Milestone 5, Student 3).

For each image: Seg-Grad-CAM on the decoder, the model's intrinsic attention map,
and a deletion/insertion + pointing-game trust score. Writes a multi-panel overlay
and a per-image JSON (so Student 4's dashboard can show "why" with a trust badge).

Run:
    py -3.14 -m src.xai.run --config configs/xai.yaml --limit 8
    py -3.14 -m src.xai.run --config configs/xai.yaml --split test
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

from ..models.infer import load_model, predict_patch
from .gradcam import grad_cam
from .attention import attention_maps
from .faithfulness import deletion_insertion, pointing_game, trust_score


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def overlay_panel(patch, cam, att, att_name, gt_mask, out_path, title, alpha):
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax[0].imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("amplitude")
    ax[1].imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(cam, cmap="jet", alpha=alpha)
    ax[1].set_title("Seg-Grad-CAM")
    ax[2].imshow(patch, cmap="gray", vmin=0, vmax=1)
    if att is not None:
        ax[2].imshow(att, cmap="jet", alpha=alpha)
    ax[2].set_title(f"attention: {att_name}")
    ax[3].imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax[3].contour(gt_mask > 0, levels=[0.5], colors="lime", linewidths=0.8)
    ax[3].set_title("defect mask (GT)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Explainable AI: Grad-CAM + attention + trust")
    ap.add_argument("--config", default="configs/xai.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    classes = cfg["classes"]
    model, _ = load_model(cfg["paths"]["checkpoint"])
    print(f"loaded {cfg['paths']['checkpoint']}")

    df = pd.read_csv(cfg["paths"]["manifest"])
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.groupby("class", group_keys=False).head(max(1, args.limit // len(classes)))
    df = df.reset_index(drop=True)

    out_dir, ov_dir = cfg["paths"]["out_dir"], cfg["paths"]["overlays_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ov_dir, exist_ok=True)
    pp = cfg["gradcam"]["method"] == "grad_cam++"
    att_key = cfg["attention"]["show"]

    rows = []
    for idx, row in df.iterrows():
        patch = np.load(row["processed_path"])
        gt = np.load(row["mask_path"]).astype(np.int64)
        true_idx = classes.index(row["class"]) + 1

        if cfg["gradcam"]["target"] == "cls":
            head, tc = "cls", classes.index(row["class"])
        elif cfg["gradcam"]["target"] == "seg_pred":
            head = "seg"
            r = predict_patch(model, patch)
            seg = r["seg"]
            tc = 2 if (seg == 2).sum() >= (seg == 1).sum() else 1
        else:                                   # seg_true
            head, tc = "seg", true_idx

        cam, info = grad_cam(model, patch, head=head, target_class=tc, plus_plus=pp)
        d_auc, i_auc = deletion_insertion(model, patch, cam, head=head, target_class=tc,
                                          steps=cfg["faithfulness"]["steps"])
        pg = pointing_game(cam, (gt == tc).astype(int))
        trust = trust_score(d_auc, i_auc, pg)

        amaps = attention_maps(model, patch)
        att = amaps.get(att_key)

        rec = {"image_id": row["image_id"], "true_class": row["class"], "split": row["split"],
               "target_class": int(tc), "method": info["method"],
               "deletion_auc": round(d_auc, 4), "insertion_auc": round(i_auc, 4),
               "pointing_hit": None if pg != pg else float(pg),
               "trust_score": round(trust, 4),
               "attention_maps": list(amaps.keys())}
        with open(os.path.join(out_dir, row["image_id"] + ".json"), "w") as f:
            json.dump(rec, f, indent=2)
        rows.append(rec)

        if idx < cfg["overlays"]["n"]:
            overlay_panel(patch, cam, att, att_key, gt,
                          os.path.join(ov_dir, row["image_id"] + ".png"),
                          f"{row['image_id']}  [{row['class']}]  {info['method']}  "
                          f"trust={trust:.2f} (del={d_auc:.2f} ins={i_auc:.2f} point={pg})",
                          cfg["overlays"]["alpha"])

    sdf = pd.DataFrame(rows)
    sdf.to_csv(os.path.join(out_dir, "_summary.csv"), index=False)
    print(f"\nDONE. explanations -> {out_dir}, overlays -> {ov_dir}")
    print(f"  images: {len(sdf)}")
    print(f"  mean trust score : {sdf['trust_score'].mean():.3f}")
    print(f"  mean deletion AUC: {sdf['deletion_auc'].mean():.3f}  (lower=better)")
    print(f"  mean insertion AUC: {sdf['insertion_auc'].mean():.3f}  (higher=better)")
    ph = sdf["pointing_hit"].dropna()
    if len(ph):
        print(f"  pointing-game hit rate: {ph.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
