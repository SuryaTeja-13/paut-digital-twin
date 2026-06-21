"""
dataset.py — PyTorch Dataset over the preprocessed patches + pseudo-masks.

Reads `manifest.csv` (Milestone 1/2 output) and serves, per image:
  image : (1, 256, 256) float32 in [0,1]
  mask  : (256, 256) long  in {0=bg, 1=porosity, 2=slag}   (segmentation target)
  label : long  in {0=porosity, 1=slag}                    (classification target)

Augmentation is TRAIN-ONLY (design_decisions.md §7/§8 — never augment val/test). The
geometric ops transform image AND mask together; intensity ops (gain, speckle,
attenuation) touch only the image, since they model PAUT acquisition physics
(architecture.md §5.9). The richer physics ops (elastic, synthetic-defect
insertion, mixup) are left as future additions for the limited-data curriculum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PAUTDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, classes=("porosity", "slag"),
                 augment: bool = False, aug_cfg: dict | None = None, limit: int | None = None,
                 fraction: float = 1.0, seed: int = 42):
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if "mask_path" not in df.columns or df["mask_path"].isna().any():
            raise ValueError("manifest has no masks — run build_pseudo_labels.py first.")
        if limit is not None:
            df = df.groupby("class", group_keys=False).head(max(1, limit // len(classes)))
        # fraction: keep a stratified-by-class fraction of this split (for the
        # data-ablation curve, architecture.md §5.11). Seeded -> reproducible.
        if fraction < 1.0:
            parts = [g.sample(frac=fraction, random_state=seed) for _, g in df.groupby("class")]
            df = pd.concat(parts).reset_index(drop=True)
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.augment = augment
        self.aug = Augmentor(aug_cfg or {}, seed) if augment else None
        # multi-defect composite augmentation (train only): blend an image with one of
        # the OTHER class so the model learns to segment BOTH flaw types in one image
        # (addresses the single-type-training limitation; architecture unchanged).
        self.composite_p = float((aug_cfg or {}).get("composite_p", 0.0)) if augment else 0.0
        self.idx_by_class = {c: self.df.index[self.df["class"] == c].tolist() for c in classes}
        self.comp_rng = np.random.default_rng(seed + 1)

    def __len__(self):
        return len(self.df)

    def _composite(self, img, mask, cur_class):
        """Max-blend with a random image of the other class; union their defect masks."""
        others = [c for c in self.classes if c != cur_class]
        pool = self.idx_by_class.get(others[0], []) if others else []
        if not pool:
            return img, mask
        j = int(self.comp_rng.choice(pool))
        row2 = self.df.iloc[j]
        img2 = np.load(row2["processed_path"]).astype(np.float32)
        mask2 = np.load(row2["mask_path"]).astype(np.int64)
        blended = np.maximum(img, img2)                       # both bright defects present
        combined = np.where(mask2 > 0, mask2, mask)           # union; each flaw keeps its class id
        return blended, combined

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.load(row["processed_path"]).astype(np.float32)      # (256,256)
        mask = np.load(row["mask_path"]).astype(np.int64)            # (256,256)
        if self.composite_p > 0 and self.comp_rng.random() < self.composite_p:
            img, mask = self._composite(img, mask, row["class"])
        if self.aug is not None:
            img, mask = self.aug(img, mask)
        img = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)   # (1,H,W)
        mask = torch.from_numpy(np.ascontiguousarray(mask))              # (H,W)
        label = torch.tensor(self.cls_to_idx[row["class"]], dtype=torch.long)
        return img, mask, label


class Augmentor:
    """Light physics-aware augmentation; each op fires with its own probability."""

    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __call__(self, img, mask):
        c = self.cfg
        # ── geometric (apply to image AND mask) ──
        if self.rng.random() < c.get("hflip_p", 0.5):
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self.rng.random() < c.get("vflip_p", 0.5):
            img, mask = img[::-1, :], mask[::-1, :]
        if self.rng.random() < c.get("rot90_p", 0.5):
            k = int(self.rng.integers(1, 4))
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        # ── intensity (image only — models gain / speckle / attenuation) ──
        if self.rng.random() < c.get("gain_p", 0.5):
            img = img * float(self.rng.uniform(*c.get("gain_range", (0.8, 1.2))))
        if self.rng.random() < c.get("speckle_p", 0.3):
            sigma = float(c.get("speckle_sigma", 0.05))
            img = img * (1.0 + self.rng.normal(0, sigma, img.shape).astype(np.float32))
        if self.rng.random() < c.get("noise_p", 0.0):
            # additive Gaussian noise — improves robustness to acquisition noise
            sigma = float(c.get("noise_sigma", 0.04))
            img = img + self.rng.normal(0, sigma, img.shape).astype(np.float32)
        if self.rng.random() < c.get("attenuation_p", 0.2):
            # depth-dependent falloff: amplitude decreases with row (depth)
            h = img.shape[0]
            falloff = np.linspace(1.0, float(c.get("attenuation_min", 0.7)), h, dtype=np.float32)
            img = img * falloff[:, None]

        return np.clip(img, 0.0, 1.0).astype(np.float32), mask
