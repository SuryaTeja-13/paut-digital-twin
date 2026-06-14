"""
blocks.py — reusable building blocks for the SCN-Attention U-Net.

Design notes tied to the architecture doc (Part 5):
  - GroupNorm, not BatchNorm: we train with tiny batches on limited data, where
    BatchNorm statistics are noisy. GroupNorm is batch-size independent (§5.2).
  - CBAM (channel + spatial attention) is the fusion attention (§5.3). Its spatial
    map doubles as a free, intrinsic explanation for Student 3.
  - Attention Gates on the skip connections suppress busy weld background (§5.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, num_groups: int = 8) -> int:
    """Largest group count <= num_groups that divides `channels` (>=1)."""
    g = min(num_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ConvBlock(nn.Module):
    """(Conv3x3 -> GroupNorm -> ReLU) x2 — the standard U-Net double conv."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch, num_groups), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_ch, num_groups), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttention(nn.Module):
    """CBAM channel attention: shared MLP over avg- and max-pooled descriptors."""

    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.mlp(x.mean(dim=(2, 3)))
        mx = self.mlp(x.amax(dim=(2, 3)))
        att = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * att


class SpatialAttention(nn.Module):
    """CBAM spatial attention: conv over channel-pooled maps -> spatial mask."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att, att        # also return the map for XAI


class CBAM(nn.Module):
    """Channel attention then spatial attention (Woo et al., 2018)."""

    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention(kernel_size)
        self.last_spatial_map = None    # cached for intrinsic explanations

    def forward(self, x):
        x = self.channel(x)
        x, sp = self.spatial(x)
        self.last_spatial_map = sp.detach()
        return x


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018). The decoder gating signal `g`
    decides which parts of the encoder skip `x` are relevant; irrelevant (busy
    background) regions are down-weighted before concatenation.
    g and x must share spatial size.
    """

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True),
                                 nn.GroupNorm(_groups(f_int), f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True),
                                 nn.GroupNorm(_groups(f_int), f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True),
                                 nn.GroupNorm(1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.last_gate_map = None

    def forward(self, g, x):
        att = self.psi(self.relu(self.w_g(g) + self.w_x(x)))
        self.last_gate_map = att.detach()
        return x * att


class UpBlock(nn.Module):
    """Up-sample (transpose conv), attention-gate the skip, concat, double-conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 num_groups: int = 8, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.gate = AttentionGate(f_g=out_ch, f_l=skip_ch, f_int=max(1, out_ch // 2))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, num_groups, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # guard against off-by-one spatial mismatch from odd sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))
