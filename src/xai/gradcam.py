"""
gradcam.py — Grad-CAM / Grad-CAM++ / Seg-Grad-CAM (architecture.md §6.1).

Chosen because a crack-segmentation study found gradient CAMs more correct,
complete and compact than gradient-free CAMs on U-Nets — and they need no
retraining and no architecture change, so zero risk to accuracy.

  - Segmentation head -> Seg-Grad-CAM on the LAST DECODER conv (`decoder_last`).
    Target score = sum of the chosen class's segmentation logits over the pixels
    the model predicts as that class (Seg-Grad-CAM).
  - Classification head -> Grad-CAM(++) on the BOTTLENECK (the classifier pools
    the bottleneck; the decoder output is NOT on its gradient path).

Returns a heatmap in [0,1] at the input resolution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_input_tensor(patch, device):
    arr = patch[0] if getattr(patch, "ndim", 2) == 3 else patch
    return torch.from_numpy(np.ascontiguousarray(arr)).float().view(1, 1, *arr.shape).to(device)


def _forward_with_target(model, x, head, target_class):
    """Run forward (grad enabled) and return (feature_map, scalar_score)."""
    out = model(x, return_features=True)
    if head == "seg":
        feats = out["decoder_last"]
        logit = out["seg"][:, target_class]                  # (1,H,W)
        pred = out["seg"].argmax(1)[0]                       # (H,W)
        mask = (pred == target_class)
        score = logit[0][mask].sum() if mask.any() else logit.sum()
    elif head == "cls":
        feats = out["bottleneck"]
        score = out["cls"][0, target_class]
    else:
        raise ValueError(head)
    return feats, score, out


def grad_cam(model, patch, head="seg", target_class=None, plus_plus=False, device=None):
    """
    Grad-CAM (or Grad-CAM++ if plus_plus=True). Returns (cam HxW float[0,1], info).
    target_class: seg class index (1=porosity,2=slag) or cls index; default =
    predicted class for cls, or class 1 for seg.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = _to_input_tensor(patch, device)

    if target_class is None:
        with torch.no_grad():
            o = model(x)
        target_class = int(o["cls"].argmax()) if head == "cls" else 1

    feats, score, _ = _forward_with_target(model, x, head, target_class)
    grads = torch.autograd.grad(score, feats, retain_graph=False)[0]   # (1,C,h,w)

    if not plus_plus:
        weights = grads.mean(dim=(2, 3), keepdim=True)                  # GAP of gradients
    else:
        g2, g3 = grads ** 2, grads ** 3
        denom = 2 * g2 + (feats * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / torch.clamp(denom, min=1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))            # (1,1,h,w)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), {"head": head, "target_class": int(target_class),
                                        "method": "grad_cam++" if plus_plus else "grad_cam"}
