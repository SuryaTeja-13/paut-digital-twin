"""
faithfulness.py — quantify whether an explanation is trustworthy (architecture.md §6.3).

Pretty heatmaps aren't enough; to align explanations with inspection standards we quantify faithfulness:

  - deletion : progressively zero out the most-important pixels (by the heatmap)
    and watch the target score fall. A faithful map -> score drops fast -> LOW
    deletion AUC.
  - insertion: start from a blank image and add the most-important pixels back;
    score should rise fast -> HIGH insertion AUC.
  - pointing game: does the heatmap's peak land inside the ground-truth mask?

We combine these into a single trust score in [0,1] the dashboard can display.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _score(model, x, head, target_class, region=None):
    out = model(x)
    if head == "seg":
        prob = F.softmax(out["seg"], dim=1)[0, target_class]
        return float(prob[region].mean()) if region is not None and region.any() else float(prob.mean())
    return float(F.softmax(out["cls"], dim=1)[0, target_class])


@torch.no_grad()
def deletion_insertion(model, patch, heatmap, head="seg", target_class=1,
                       steps=20, device=None):
    """
    Returns (deletion_auc, insertion_auc), each in [0,1], normalized by the
    full-image score. Lower deletion + higher insertion = more faithful.

    For segmentation the score is the target-class probability averaged over the
    originally-predicted defect REGION (so the tiny defect isn't swamped by the
    huge background).
    """
    device = device or next(model.parameters()).device
    arr = patch[0] if getattr(patch, "ndim", 2) == 3 else patch
    base = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)
    h, w = base.shape
    order = np.ascontiguousarray(np.argsort(heatmap.ravel())[::-1])   # most important first
    n = h * w
    chunk = max(1, n // steps)

    region = None
    if head == "seg":
        with torch.no_grad():
            pred = model(base.view(1, 1, h, w))["seg"].argmax(1)[0]
        region = (pred == target_class)
        if not region.any():
            region = None

    def run(img):
        return _score(model, img.view(1, 1, h, w), head, target_class, region)

    # deletion: from full image, remove top pixels
    img = base.clone()
    del_scores = [run(img)]
    flat = img.view(-1)
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        flat[idx] = 0.0
        del_scores.append(run(img))

    # insertion: from blank, add top pixels
    img = torch.zeros_like(base)
    ins_scores = [run(img)]
    flat, src = img.view(-1), base.view(-1)
    for i in range(steps):
        idx = order[i * chunk:(i + 1) * chunk]
        flat[idx] = src[idx]
        ins_scores.append(run(img))

    # normalize by the full-image score so both AUCs are in [0,1]
    ref = max(del_scores[0], 1e-8)
    del_auc = float(np.clip(np.mean(del_scores) / ref, 0.0, 1.0))
    ins_auc = float(np.clip(np.mean(ins_scores) / ref, 0.0, 1.0))
    return del_auc, ins_auc


def pointing_game(heatmap, gt_mask):
    """1.0 if the heatmap's peak falls inside the ground-truth defect mask, else 0.0."""
    if gt_mask.sum() == 0:
        return float("nan")
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    return float(bool(gt_mask[peak] > 0))


def trust_score(deletion_auc, insertion_auc, point_hit):
    """
    Single [0,1] trust score: faithful = low deletion, high insertion, peak on
    target. point_hit may be NaN (no GT) -> dropped from the average.
    """
    parts = [insertion_auc, 1.0 - deletion_auc]
    if point_hit == point_hit:        # not NaN
        parts.append(point_hit)
    return float(np.clip(np.mean(parts), 0.0, 1.0))
