"""
attention.py — surface the model's INTRINSIC attention maps (architecture.md §6.1).

The model already produces attention internally — CBAM spatial-attention maps in
each fusion block and Attention-Gate coefficients on each decoder skip. These are
faithful by construction and free (no extra computation), so we expose them as
first-class explanations alongside Grad-CAM.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _up(map_2d, size):
    m = map_2d.float().view(1, 1, *map_2d.shape[-2:])
    m = F.interpolate(m, size=size, mode="bilinear", align_corners=False)[0, 0]
    m = m - m.min()
    return (m / (m.max() + 1e-8)).cpu().numpy()


@torch.no_grad()
def attention_maps(model, patch, device=None):
    """
    Run a forward pass and collect the cached CBAM + Attention-Gate maps,
    each upsampled to the input size and normalised to [0,1].
    Returns {name: HxW array}.
    """
    device = device or next(model.parameters()).device
    arr = patch[0] if getattr(patch, "ndim", 2) == 3 else patch
    x = torch.from_numpy(np.ascontiguousarray(arr)).float().view(1, 1, *arr.shape).to(device)
    size = x.shape[-2:]
    model.eval()
    model(x)                                  # populates the cached maps

    maps = {}
    if getattr(model, "use_scattering", False):
        for name in ["fuse1", "fuse2", "fuse3", "fuse4", "fuseb"]:
            block = getattr(model, name, None)
            if block is not None and block.cbam.last_spatial_map is not None:
                maps[f"cbam_{name}"] = _up(block.cbam.last_spatial_map[0, 0], size)
    for name in ["up1", "up2", "up3", "up4"]:
        block = getattr(model, name, None)
        if block is not None and block.gate.last_gate_map is not None:
            maps[f"gate_{name}"] = _up(block.gate.last_gate_map[0, 0], size)
    return maps
