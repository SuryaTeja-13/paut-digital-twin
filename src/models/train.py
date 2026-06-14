"""
train.py — train the SCN-Attention U-Net (Student 2).

Device-aware (CPU or CUDA, auto-detected), fully config-driven. The same command
runs a tiny CPU smoke-test locally and a full GPU run on Colab/Kaggle.

Examples:
    # full training (use on GPU)
    py -3.14 -m src.models.train --config configs/model.yaml

    # CPU smoke-test: tiny subset, few epochs, just prove the pipeline runs
    py -3.14 -m src.models.train --config configs/model.yaml --limit 12 --epochs 2

    # prove the model can LEARN: overfit ~8 images, watch train Dice -> ~1.0
    py -3.14 -m src.models.train --config configs/model.yaml --overfit 8 --epochs 40

    # plain U-Net baseline for the ablation (no scattering)
    py -3.14 -m src.models.train --config configs/model.yaml --no-scattering
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import PAUTDataset
from .scn_attention_unet import build_model
from .losses import MultiTaskLoss
from .metrics import SegAccumulator


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(epoch, cfg):
    """Cosine schedule with linear warmup, returns a multiplier in (0,1]."""
    t = cfg["train"]
    warm, total = t.get("warmup_epochs", 0), t["epochs"]
    if t.get("scheduler", "cosine") != "cosine":
        return 1.0
    if epoch < warm:
        return (epoch + 1) / max(1, warm)
    prog = (epoch - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(model, loader, criterion, device, seg_classes):
    model.eval()
    acc = SegAccumulator(seg_classes)
    correct = total = 0
    loss_sum = 0.0
    for img, mask, label in loader:
        img, mask, label = img.to(device), mask.to(device), label.to(device)
        out = model(img)
        loss, _ = criterion(out, mask, label)
        loss_sum += float(loss) * img.size(0)
        acc.update(out["seg"].argmax(1), mask)
        correct += int((out["cls"].argmax(1) == label).sum())
        total += img.size(0)
    dice, iou = acc.finalize()
    return {
        "loss": loss_sum / max(1, total),
        "dice": dice, "iou": iou,
        "dice_fg": float(np.mean(dice[1:])),     # mean over defect classes
        "cls_acc": correct / max(1, total),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train SCN-Attention U-Net")
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap train/val images (smoke-test)")
    ap.add_argument("--overfit", type=int, default=None,
                    help="overfit N train images (no aug, no val) to prove the model learns")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs")
    ap.add_argument("--no-scattering", action="store_true", help="plain U-Net baseline")
    ap.add_argument("--data-fraction", type=float, default=1.0,
                    help="train on this fraction of the TRAIN split (val/test stay full) — for §5.11 ablation")
    ap.add_argument("--ckpt-dir", default=None,
                    help="override checkpoint dir (ablation uses a separate dir so it never "
                         "overwrites the production checkpoint)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.no_scattering:
        cfg["model"]["use_scattering"] = False
    if args.overfit is not None:
        # memorization test: constant LR (no decay) and no dropout, so the model
        # is free to memorize the tiny set — this checks the model CAN learn.
        cfg["train"]["scheduler"] = "none"
        cfg["train"]["warmup_epochs"] = 0
        cfg["model"]["dropout"] = 0.0
    set_seed(cfg["seed"])
    device = get_device()

    seg_classes = cfg["model"]["seg_classes"]
    classes = cfg["data"]["classes"]
    manifest = cfg["data"]["manifest"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"].get("num_workers", 0)

    print(f"device={device}  scattering={cfg['model']['use_scattering']}  "
          f"batch={bs}  epochs={cfg['train']['epochs']}")

    # ── data ──
    overfit = args.overfit is not None
    if overfit:
        train_ds = PAUTDataset(manifest, "train", classes, augment=False, limit=args.overfit,
                               seed=cfg["seed"])
        val_ds = None
        print(f"OVERFIT mode: {len(train_ds)} train images, no augmentation, no val")
    else:
        train_ds = PAUTDataset(manifest, "train", classes,
                               augment=cfg["augment"].get("enabled", True),
                               aug_cfg=cfg["augment"], limit=args.limit,
                               fraction=args.data_fraction, seed=cfg["seed"])
        val_ds = PAUTDataset(manifest, "val", classes, augment=False, limit=args.limit,
                             seed=cfg["seed"])
        if args.data_fraction < 1.0:
            print(f"DATA-ABLATION: training on {args.data_fraction:.0%} of train = {len(train_ds)} images")
        print(f"train={len(train_ds)}  val={len(val_ds)}")

    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False)
    val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw) if val_ds else None

    # ── model / loss / optim ──
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.2f}M")
    criterion = MultiTaskLoss(seg_classes=seg_classes, **cfg["loss"])
    # class weighting for the imbalanced classifier (269 porosity / 525 slag).
    if cfg["train"].get("class_weighting", True) and not overfit:
        labels = [train_ds.cls_to_idx[c] for c in train_ds.df["class"]]
        counts = np.bincount(labels, minlength=len(classes)).astype(np.float32)
        weights = counts.sum() / (len(classes) * np.clip(counts, 1, None))
        criterion.cls_weight = torch.tensor(weights, dtype=torch.float32)
        print(f"class counts {counts.tolist()} -> cls weights {np.round(weights,3).tolist()}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    base_lr = cfg["train"]["lr"]
    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = args.ckpt_dir or cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = cfg["train"]["ckpt_name"] + ("" if cfg["model"]["use_scattering"] else "_baseline")
    if args.data_fraction < 1.0:                       # keep ablation runs separate
        tag += f"_frac{int(round(args.data_fraction * 100))}"
    best_path = os.path.join(ckpt_dir, tag + "_best.pt")

    best_metric, best_cls_acc, since_improve = -1.0, 0.0, 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        mult = lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = base_lr * mult

        t0 = time.time()
        run = {}
        for img, mask, label in train_ld:
            img, mask, label = img.to(device), mask.to(device), label.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(img)
                loss, parts = criterion(out, mask, label)
            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                run[k] = run.get(k, 0.0) + v
        run = {k: v / len(train_ld) for k, v in run.items()}
        dt = time.time() - t0

        msg = (f"epoch {epoch+1:>3}/{cfg['train']['epochs']}  lr={base_lr*mult:.2e}  "
               f"loss={run['total']:.4f} (seg={run['seg']:.3f} tversky={run['tversky']:.3f} "
               f"cls={run['cls']:.3f} bnd={run['boundary']:.3f})  {dt:.1f}s")

        if val_ld is not None:
            val = evaluate(model, val_ld, criterion, device, seg_classes)
            msg += (f"  | val dice_fg={val['dice_fg']:.3f} "
                    f"(por={val['dice'][1]:.3f} slag={val['dice'][2]:.3f}) "
                    f"cls_acc={val['cls_acc']:.3f}")
            metric = val["dice_fg"]
            if metric > best_metric:
                best_metric, best_cls_acc, since_improve = metric, val["cls_acc"], 0
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                            "val_dice_fg": metric}, best_path)
                msg += "  *best"
            else:
                since_improve += 1
        else:
            # overfit mode: report TRAIN dice so we can see it learn
            tr = evaluate(model, train_ld, criterion, device, seg_classes)
            msg += f"  | train dice_fg={tr['dice_fg']:.3f} cls_acc={tr['cls_acc']:.3f}"

        print(msg)
        if val_ld is not None and since_improve >= cfg["train"].get("early_stop_patience", 1e9):
            print(f"early stop: no val improvement in {since_improve} epochs")
            break

    if val_ld is not None:
        print(f"\nbest val dice_fg = {best_metric:.4f}  ->  {best_path}")
        # log metrics next to the checkpoint (the ablation runner reads these)
        metrics = {"data_fraction": args.data_fraction,
                   "use_scattering": cfg["model"]["use_scattering"],
                   "n_train": len(train_ds), "best_val_dice_fg": best_metric,
                   "best_val_cls_acc": best_cls_acc}
        with open(best_path.replace("_best.pt", "_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    print("\noverfit run complete (no checkpoint saved in overfit mode)")
    return {}


if __name__ == "__main__":
    main()
    sys.exit(0)
