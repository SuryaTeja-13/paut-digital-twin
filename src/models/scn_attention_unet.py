"""
scn_attention_unet.py — the core multi-task model (Student 2, architecture.md Part 5).

Two branches from the 1x256x256 amplitude patch:
  (a) Attention U-Net encoder  -> E1..E4, bottleneck
  (b) fixed Wavelet Scattering -> one scattering tensor S + a global vector

Fused at every scale:  concat(E_i, resize(1x1Conv(S))) -> 1x1Conv -> CBAM.
Decoded with attention-gated skips. Three heads:
  Head 1 segmentation  (background / porosity / slag)          -> per-pixel logits
  Head 2 classification (porosity vs slag)                     -> image-level logits
  Head 3 size regression                                       -> OPTIONAL, off by default
                                                                  (size is derived from the mask)

Set use_scattering=False to get a plain Attention-U-Net baseline for the
data-ablation comparison (architecture.md §5.11).

Everything is GPU-aware but CPU-runnable; device is chosen by the caller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, CBAM, UpBlock
from .scattering import ScatteringBranch


class SCNFusion(nn.Module):
    """Fuse encoder features at one scale with the scattering tensor (§5.3)."""

    def __init__(self, enc_ch: int, scat_ch: int, num_groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(scat_ch, enc_ch, kernel_size=1)     # SCN coeff -> Ci channels
        self.mix = nn.Sequential(
            nn.Conv2d(2 * enc_ch, enc_ch, kernel_size=1, bias=False),
            nn.GroupNorm(min(num_groups, enc_ch), enc_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(enc_ch)

    def forward(self, enc_feat, scat):
        s = self.proj(scat)
        s = F.interpolate(s, size=enc_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.cbam(self.mix(torch.cat([enc_feat, s], dim=1)))


class SCNAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        seg_classes: int = 3,        # background, porosity, slag (decision §3)
        n_classes: int = 2,          # porosity vs slag (decision §2)
        base: int = 32,              # encoder width: 32->64->128->256, bottleneck 512
        use_scattering: bool = True,
        scattering: dict | None = None,
        dropout: float = 0.1,
        num_groups: int = 8,
        size_head: bool = False,
    ):
        super().__init__()
        self.use_scattering = use_scattering
        self.size_head_enabled = size_head
        c1, c2, c3, c4, cb = base, base * 2, base * 4, base * 8, base * 16

        # ── encoder ──
        self.enc1 = ConvBlock(in_channels, c1, num_groups, dropout)
        self.enc2 = ConvBlock(c1, c2, num_groups, dropout)
        self.enc3 = ConvBlock(c2, c3, num_groups, dropout)
        self.enc4 = ConvBlock(c3, c4, num_groups, dropout)
        self.bottleneck = ConvBlock(c4, cb, num_groups, dropout)
        self.pool = nn.MaxPool2d(2)

        # ── scattering branch + per-scale fusion ──
        scat_dim = 0
        if use_scattering:
            scattering = scattering or {}
            self.scat = ScatteringBranch(
                J=scattering.get("J", 2), L=scattering.get("L", 8),
                shape=tuple(scattering.get("shape", (256, 256))),
                max_order=scattering.get("order", 2),
            )
            scat_dim = self.scat.out_channels
            self.fuse1 = SCNFusion(c1, scat_dim, num_groups)
            self.fuse2 = SCNFusion(c2, scat_dim, num_groups)
            self.fuse3 = SCNFusion(c3, scat_dim, num_groups)
            self.fuse4 = SCNFusion(c4, scat_dim, num_groups)
            self.fuseb = SCNFusion(cb, scat_dim, num_groups)

        # ── decoder (attention-gated skips) ──
        self.up4 = UpBlock(cb, c4, c4, num_groups, dropout)
        self.up3 = UpBlock(c4, c3, c3, num_groups, dropout)
        self.up2 = UpBlock(c3, c2, c2, num_groups, dropout)
        self.up1 = UpBlock(c2, c1, c1, num_groups, dropout)

        # ── heads ──
        self.seg_head = nn.Conv2d(c1, seg_classes, kernel_size=1)   # last decoder conv (XAI target)
        cls_in = cb + scat_dim                                       # bottleneck pool + global SCN vec
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, n_classes),
        )
        if size_head:
            self.size_head = nn.Sequential(nn.Linear(cls_in, 64), nn.ReLU(inplace=True),
                                           nn.Linear(64, 1))

    def forward(self, x, return_features: bool = False):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        scat_vec = None
        if self.use_scattering:
            s = self.scat(x)                       # (B, scat_ch, h, w)
            scat_vec = s.mean(dim=(2, 3))          # global scattering vector
            e1 = self.fuse1(e1, s)
            e2 = self.fuse2(e2, s)
            e3 = self.fuse3(e3, s)
            e4 = self.fuse4(e4, s)
            b = self.fuseb(b, s)

        # decoder
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        seg_logits = self.seg_head(d1)

        pooled = b.mean(dim=(2, 3))
        cls_in = torch.cat([pooled, scat_vec], dim=1) if scat_vec is not None else pooled
        cls_logits = self.cls_head(cls_in)

        out = {"seg": seg_logits, "cls": cls_logits}
        if self.size_head_enabled:
            out["size"] = self.size_head(cls_in)
        if return_features:
            out["decoder_last"] = d1               # Seg-Grad-CAM target (segmentation head)
            out["bottleneck"] = b                  # Grad-CAM target (classification head)
        return out


def build_model(cfg: dict) -> SCNAttentionUNet:
    """Construct the model from a config dict (configs/model.yaml -> model:)."""
    m = cfg.get("model", cfg)
    return SCNAttentionUNet(
        in_channels=m.get("in_channels", 1),
        seg_classes=m.get("seg_classes", 3),
        n_classes=m.get("n_classes", 2),
        base=m.get("base", 32),
        use_scattering=m.get("use_scattering", True),
        scattering=m.get("scattering", {}),
        dropout=m.get("dropout", 0.1),
        num_groups=m.get("num_groups", 8),
        size_head=m.get("size_head", False),
    )
