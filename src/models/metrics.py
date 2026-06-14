"""
metrics.py — evaluation metrics (architecture.md §5.12).

Segmentation: per-class Dice and IoU (hard, from argmax predictions).
Classification: accuracy.
These are reported on the validation split — the honest accuracy meter.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def seg_dice_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
    """
    pred, target: (B, H, W) long. Returns dicts {class_idx: dice}, {class_idx: iou}
    accumulated as (intersection, |a|, |b|) sums — call accumulate() then finalize.
    """
    dice, iou = {}, {}
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        psum, tsum = p.sum().float(), t.sum().float()
        union = psum + tsum - inter
        dice[c] = float((2 * inter + eps) / (psum + tsum + eps))
        iou[c] = float((inter + eps) / (union + eps))
    return dice, iou


class SegAccumulator:
    """Accumulate intersection / set sizes over batches for unbiased epoch metrics."""

    def __init__(self, num_classes: int):
        self.n = num_classes
        self.inter = torch.zeros(num_classes)
        self.psum = torch.zeros(num_classes)
        self.tsum = torch.zeros(num_classes)

    @torch.no_grad()
    def update(self, pred, target):
        for c in range(self.n):
            p = (pred == c)
            t = (target == c)
            self.inter[c] += (p & t).sum().cpu()
            self.psum[c] += p.sum().cpu()
            self.tsum[c] += t.sum().cpu()

    def finalize(self, eps: float = 1e-6):
        dice = (2 * self.inter + eps) / (self.psum + self.tsum + eps)
        iou = (self.inter + eps) / (self.psum + self.tsum - self.inter + eps)
        return dice.tolist(), iou.tolist()
