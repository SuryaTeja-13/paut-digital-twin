"""
evaluate.py — full evaluation of a trained checkpoint (architecture.md §5.12).

Reports, on a chosen split:
  - segmentation: per-class Dice and IoU (background / porosity / slag), mean foreground Dice
  - classification: accuracy, BALANCED accuracy (data is imbalanced), confusion matrix
  - "no-defect" rate: fraction of images whose predicted mask is essentially empty

Run:
    py -3.14 -m src.models.evaluate --ckpt checkpoints/scn_attn_unet_best.pt --split val
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd

from .infer import load_model, predict_patch
from .metrics import SegAccumulator


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained SCN-Attention U-Net")
    ap.add_argument("--ckpt", default="checkpoints/scn_attn_unet_best.pt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--save", default=None, help="write metrics JSON here (dashboard reads it)")
    args = ap.parse_args(argv)

    model, cfg = load_model(args.ckpt)
    classes = cfg["data"]["classes"]
    seg_classes = cfg["model"]["seg_classes"]
    seg_names = ["background"] + list(classes)

    df = pd.read_csv(args.manifest)
    sub = df[df["split"] == args.split].reset_index(drop=True)
    print(f"evaluating {len(sub)} {args.split} images from {args.ckpt}\n")

    acc = SegAccumulator(seg_classes)
    import torch
    y_true, y_pred, y_pred_seg = [], [], []
    empty = 0
    for _, row in sub.iterrows():
        patch = np.load(row["processed_path"])
        r = predict_patch(model, patch)
        acc.update(torch.from_numpy(r["seg"]).unsqueeze(0),
                   torch.from_numpy(np.load(row["mask_path"]).astype(np.int64)).unsqueeze(0))
        y_true.append(classes.index(row["class"]))
        y_pred.append(r["cls_idx"])
        # seg-derived type = which defect class-channel has more pixels (a
        # baseline comparison; the deployed app uses the scattering-feature
        # classifier below)
        areas = [int((r["seg"] == ci + 1).sum()) for ci in range(len(classes))]
        y_pred_seg.append(int(np.argmax(areas)))
        if int((r["seg"] > 0).sum()) < 10:        # near-empty predicted mask
            empty += 1

    dice, iou = acc.finalize()
    y_true, y_pred, y_pred_seg = np.array(y_true), np.array(y_pred), np.array(y_pred_seg)

    print("== Segmentation (per class) ==")
    for i, name in enumerate(seg_names):
        print(f"  {name:<11} Dice={dice[i]:.3f}  IoU={iou[i]:.3f}")
    print(f"  mean foreground Dice = {np.mean(dice[1:]):.3f}\n")

    print("== Classification (porosity vs slag) ==")
    overall = float((y_true == y_pred).mean())
    per_class = []
    for c, name in enumerate(classes):
        m = y_true == c
        a = float((y_pred[m] == c).mean()) if m.any() else float("nan")
        per_class.append(a)
        print(f"  {name:<9} acc = {a:.3f}  (n={int(m.sum())})")
    print(f"  overall acc          = {overall:.3f}")
    print(f"  BALANCED acc         = {np.nanmean(per_class):.3f}   (0.5 = chance)")
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"  confusion [rows=true {classes}, cols=pred]:\n{cm}\n")

    # seg-derived type accuracy (what the dashboard uses)
    print("== Type via SEGMENTATION (baseline) ==")
    seg_per_class = []
    for c, name in enumerate(classes):
        m = y_true == c
        a = float((y_pred_seg[m] == c).mean()) if m.any() else float("nan")
        seg_per_class.append(a)
        print(f"  {name:<9} acc = {a:.3f}  (n={int(m.sum())})")
    seg_balanced = float(np.nanmean(seg_per_class))
    print(f"  BALANCED acc         = {seg_balanced:.3f}")

    # scattering-feature type classifier (what the app actually uses)
    scat_clf_metrics = None
    tcpath = "checkpoints/type_classifier.pkl"
    if os.path.exists(tcpath):
        from .type_classifier import TypeClassifier, scatter_features
        tclf = TypeClassifier.load(tcpath)
        feats = scatter_features(model.scat,
                                 np.stack([np.load(p) for p in sub["processed_path"]]))
        sc_pred, _ = tclf.predict_features(feats)
        print("\n== Type via SCATTERING CLASSIFIER (used by the app) ==")
        sc_per = []
        for c, name in enumerate(classes):
            m = y_true == c
            a = float((sc_pred[m] == c).mean()) if m.any() else float("nan")
            sc_per.append(a)
            print(f"  {name:<9} acc = {a:.3f}  (n={int(m.sum())})")
        sc_bal = float(np.nanmean(sc_per))
        print(f"  BALANCED acc         = {sc_bal:.3f}")
        scat_clf_metrics = {"balanced_acc": round(sc_bal, 4),
                            "per_class": {classes[c]: round(sc_per[c], 4) for c in range(len(classes))}}

    print(f"\n== 'No-defect' (near-empty predicted mask): {empty}/{len(sub)} images ==")

    if args.save:
        import json
        metrics = {
            "checkpoint": os.path.basename(args.ckpt), "split": args.split,
            "n_images": int(len(sub)),
            "segmentation": {"dice": {seg_names[i]: round(dice[i], 4) for i in range(seg_classes)},
                             "iou": {seg_names[i]: round(iou[i], 4) for i in range(seg_classes)},
                             "mean_fg_dice": round(float(np.mean(dice[1:])), 4)},
            # PRIMARY type signal the app uses (scattering classifier), if trained
            "type_via_scattering_clf": scat_clf_metrics,
            "type_via_segmentation": {
                "balanced_acc": round(seg_balanced, 4),
                "per_class": {classes[c]: round(seg_per_class[c], 4) for c in range(len(classes))}},
            "type_via_classifier_head": {  # diagnostic only (weaker neural head)
                "balanced_acc": round(float(np.nanmean(per_class)), 4),
                "overall_acc": round(overall, 4)},
            "no_defect_rate": round(empty / max(len(sub), 1), 4),
        }
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nmetrics saved -> {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
