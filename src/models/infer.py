"""
infer.py — load a trained checkpoint and predict masks/classes (CPU or GPU).

This is the S2 -> S3/S4 hand-off: given an image (raw file OR a preprocessed
patch), return the predicted segmentation label map, the class probabilities, and
(optionally) the decoder features / attention maps that Student 3's XAI needs.

The checkpoint stores the model config, so we rebuild the exact architecture and
load the weights — no need to know the hyperparameters here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .scn_attention_unet import build_model
from ..data.preprocess import process_image


def get_device(prefer_cuda: bool = True):
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def load_model(ckpt_path: str, device=None):
    """Rebuild the model from the checkpoint's saved config and load weights."""
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("cfg", {})


@torch.no_grad()
def predict_patch(model, patch: np.ndarray, device=None, return_features: bool = False):
    """
    patch: (H,W) or (1,H,W) float32 in [0,1] (already preprocessed).
    Returns dict: seg (H,W int), seg_prob (C,H,W), cls_idx, cls_prob, [features].
    """
    device = device or next(model.parameters()).device
    arr = patch[0] if patch.ndim == 3 else patch
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().view(1, 1, *arr.shape).to(device)
    out = model(x, return_features=return_features)
    seg_prob = F.softmax(out["seg"], dim=1)[0]
    cls_prob = F.softmax(out["cls"], dim=1)[0]
    res = {
        "seg": seg_prob.argmax(0).cpu().numpy().astype(np.uint8),
        "seg_prob": seg_prob.cpu().numpy(),
        "cls_idx": int(cls_prob.argmax()),
        "cls_prob": cls_prob.cpu().numpy(),
    }
    if return_features and "decoder_last" in out:
        res["decoder_last"] = out["decoder_last"]
    return res


def predict_image(model, image_path: str, preprocess_cfg: dict, device=None,
                  return_features: bool = False):
    """Raw image file -> preprocess to a patch -> predict. Returns (result, patch, meta)."""
    patch, meta = process_image(image_path, preprocess_cfg)
    res = predict_patch(model, patch, device, return_features)
    return res, patch, meta
