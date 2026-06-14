"""
scattering.py — the fixed Wavelet Scattering branch (SCN), via Kymatio.

WHY this branch exists (the whole reason for the project, architecture.md Part 2):
a wavelet scattering transform uses FIXED filters — no training, so it cannot
overfit our tiny PAUT dataset — and produces translation-invariant, deformation-
stable, speckle-robust features. That prior is what lets the model generalize
from few noisy ultrasonic images.

The scattering filters are NOT trained (design_decisions.md decision §9). Only the small
1x1 "projection" convs that adapt the scattering tensor to each U-Net scale are
learnable — those are part of the fusion, not the scattering itself.

Kymatio note: `from kymatio.torch import ...` eagerly imports a 3D module that
needs scipy.special.sph_harm, which scipy 1.17 removed. We import the 2D torch
frontend directly to avoid that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D


class ScatteringBranch(nn.Module):
    """
    Wraps Kymatio Scattering2D for a 1-channel input.

    forward(x): (B,1,H,W) -> (B, C, H/2^J, W/2^J), with the scattering coefficient
    channels flattened into C. `out_channels` and `out_size` are filled in lazily
    on the first forward (they depend on J/L and input size).
    """

    def __init__(self, J: int = 2, L: int = 8, shape=(256, 256), max_order: int = 2):
        super().__init__()
        self.J, self.L, self.shape, self.max_order = J, L, tuple(shape), max_order
        self.scattering = ScatteringTorch2D(J=J, L=L, shape=tuple(shape), max_order=max_order)
        # freeze: scattering carries fixed (non-learnable) filter buffers, but make
        # the intent explicit so nothing ever updates them.
        for p in self.scattering.parameters():
            p.requires_grad_(False)
        self.out_channels = self._coeff_count(J, L, max_order)
        self.out_size = (shape[0] // (2 ** J), shape[1] // (2 ** J))

    @staticmethod
    def _coeff_count(J: int, L: int, order: int) -> int:
        c = 1                       # order 0 (low-pass)
        if order >= 1:
            c += J * L              # order 1
        if order >= 2:
            c += (L * L) * (J * (J - 1)) // 2   # order 2
        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # filters are fixed -> no autograd graph needed through the transform
        with torch.no_grad():
            s = self.scattering(x)              # (B, 1, C, h, w)
        b, _, c, h, w = s.shape
        return s.reshape(b, c, h, w)
