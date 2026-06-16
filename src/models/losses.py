"""
losses.py — the multi-task loss (architecture.md §5.6 + §9, design_decisions.md decision §10).

    L = w_seg * L_seg + w_cls * CE(classification) + w_boundary * boundary

with  L_seg = dice_w * Dice + focal_w * FocalCE + tversky_w * FocalTversky.

WHY Tversky / foreground-focused Dice: defects are ~0.15% of pixels. A plain
Dice averaged over all classes (incl. background) lets the model score ~0.66 by
predicting ALL background and abandoning the defect — we saw exactly this
collapse. So:
  - Dice and Tversky are averaged over FOREGROUND classes only (background
    excluded), so nailing background gives no free credit.
  - Tversky with beta > alpha penalizes FALSE NEGATIVES (missed defect pixels)
    harder than false positives — thin/faint defects fail by being missed.
  - Focal-Tversky (gamma > 1) further focuses on hard, still-wrong regions.

All weights live in configs/model.yaml so every tuning change is logged (§10).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fg_slice(x, include_bg):
    """Channels to average over: all, or foreground only (drop class 0 = bg)."""
    return x if include_bg else x[:, 1:]


def soft_dice_loss(probs, target_onehot, include_bg=False, eps=1e-6):
    """1 - mean per-class soft Dice over the selected channels. (B,C,H,W)."""
    p, t = _fg_slice(probs, include_bg), _fg_slice(target_onehot, include_bg)
    dims = (0, 2, 3)
    inter = (p * t).sum(dims)
    denom = p.sum(dims) + t.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_tversky_loss(probs, target_onehot, alpha=0.3, beta=0.7, gamma=1.0,
                       include_bg=False, eps=1e-6):
    """
    Focal-Tversky over the selected channels.
    alpha weights false positives, beta weights false negatives (beta>alpha =>
    misses hurt more). gamma>1 focuses on hard cases (gamma=1 => plain Tversky).
    """
    p, t = _fg_slice(probs, include_bg), _fg_slice(target_onehot, include_bg)
    dims = (0, 2, 3)
    tp = (p * t).sum(dims)
    fp = (p * (1 - t)).sum(dims)
    fn = ((1 - p) * t).sum(dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1.0 - tversky) ** gamma).mean()


def focal_ce_loss(logits, target, gamma=2.0):
    """Per-pixel focal cross-entropy. logits (B,C,H,W), target (B,H,W) long."""
    logp = F.log_softmax(logits, dim=1)
    ce = F.nll_loss(logp, target, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def boundary_loss(probs, target_onehot):
    """L1 between EDGES of predicted vs target foreground prob (differentiable)."""
    fg_p = 1.0 - probs[:, :1]
    fg_t = 1.0 - target_onehot[:, :1]

    def edge(m):
        dil = F.max_pool2d(m, 3, stride=1, padding=1)
        ero = -F.max_pool2d(-m, 3, stride=1, padding=1)
        return dil - ero

    return F.l1_loss(edge(fg_p), edge(fg_t))


class MultiTaskLoss(nn.Module):
    def __init__(self, seg_classes=3, w_seg=0.6, w_cls=0.3, w_boundary=0.1,
                 focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0,
                 tversky_weight=1.0, tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_gamma=1.0, include_background_in_dice=False):
        super().__init__()
        self.seg_classes = seg_classes
        self.w_seg, self.w_cls, self.w_boundary = w_seg, w_cls, w_boundary
        self.focal_gamma = focal_gamma
        self.dice_weight, self.focal_weight = dice_weight, focal_weight
        self.tversky_weight = tversky_weight
        self.t_alpha, self.t_beta, self.t_gamma = tversky_alpha, tversky_beta, tversky_gamma
        self.include_bg = include_background_in_dice
        # set by the trainer; a safeguard for class imbalance (dataset is 525/525
        # balanced now). a (n_classes,) tensor weighting the classification cross-entropy.
        self.cls_weight = None

    def forward(self, out, mask, cls_target):
        seg_logits = out["seg"]
        probs = F.softmax(seg_logits, dim=1)
        onehot = F.one_hot(mask.long(), self.seg_classes).permute(0, 3, 1, 2).float()

        dice = soft_dice_loss(probs, onehot, self.include_bg)
        focal = focal_ce_loss(seg_logits, mask.long(), self.focal_gamma)
        tversky = focal_tversky_loss(probs, onehot, self.t_alpha, self.t_beta,
                                     self.t_gamma, self.include_bg)
        seg = (self.dice_weight * dice + self.focal_weight * focal
               + self.tversky_weight * tversky)
        bnd = boundary_loss(probs, onehot)
        w = self.cls_weight.to(out["cls"].device) if self.cls_weight is not None else None
        cls = F.cross_entropy(out["cls"], cls_target.long(), weight=w)

        total = self.w_seg * seg + self.w_cls * cls + self.w_boundary * bnd
        return total, {
            "total": float(total.detach()),
            "seg": float(seg.detach()),
            "dice": float(dice.detach()),
            "focal": float(focal.detach()),
            "tversky": float(tversky.detach()),
            "boundary": float(bnd.detach()),
            "cls": float(cls.detach()),
        }
