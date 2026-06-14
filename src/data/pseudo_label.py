"""
pseudo_label.py — Stage-1 pseudo-mask generation (architecture.md §4.3, §5 Stage 1).

The model can't train without segmentation targets, and we have no hand-drawn
masks yet. So we bootstrap APPROXIMATE masks straight from the amplitude with the
classic recipe:

    (optional smooth) -> threshold -> morphology (open/close/fill)
                      -> drop tiny specks -> label defect pixels with the class

Defects are bright on a near-black background, so Otsu cleanly separates them.
Each image is single-class (porosity OR slag), so every defect pixel in that
image gets that one class index:

    0 = background,  1 = porosity,  2 = slag      (design_decisions.md decision §3)

These masks are a STARTING POINT, not ground truth — that is the whole point of
Stage 1. Pure functions here; disk I/O lives in generate() / the CLI.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label


def class_index_map(classes: list[str]) -> dict[str, int]:
    """porosity->1, slag->2, ... (0 reserved for background)."""
    return {c: i + 1 for i, c in enumerate(classes)}


def _drop_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only connected components with at least `min_area` pixels."""
    lab = cc_label(mask)
    if lab.max() == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_area)[0]
    keep = keep[keep != 0]                  # 0 is background
    return np.isin(lab, keep)


# ───────────────────────── binary defect detection ───────────────────────

def compute_threshold(patch: np.ndarray, thr_cfg: dict) -> float:
    """Pick the amplitude cut-off separating defect from background."""
    method = thr_cfg.get("method", "otsu")
    if method == "otsu":
        # Otsu on a histogram that is ~99% zeros lands in the gap above the
        # background mass — exactly the bright-defect boundary we want.
        try:
            t = float(threshold_otsu(patch))
        except Exception:
            t = float(thr_cfg.get("fixed_value", 0.20))
    elif method == "percentile":
        t = float(np.percentile(patch, thr_cfg.get("percentile", 99.0)))
    elif method == "fixed":
        t = float(thr_cfg.get("fixed_value", 0.20))
    else:
        raise ValueError(f"unknown threshold method: {method}")
    t = t + float(thr_cfg.get("offset", 0.0))
    return max(t, float(thr_cfg.get("floor", 0.08)))


def binary_defect_mask(patch: np.ndarray, cfg: dict) -> np.ndarray:
    """Amplitude patch -> cleaned boolean defect mask."""
    img = patch.astype(np.float32)

    sigma = float(cfg.get("smooth", {}).get("sigma", 0.0))
    if sigma > 0:
        img = ndi.gaussian_filter(img, sigma=sigma)

    t = compute_threshold(img, cfg["threshold"])
    mask = img > t

    # Use OpenCV morphology: it is a correct (extensive) closing. scipy.ndimage's
    # binary_closing over-erodes thin defects here, shrinking masks ~3x.
    m = cfg.get("morphology", {})
    u8 = mask.astype(np.uint8)
    ok = int(m.get("open_ksize", 0))
    if ok > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kern)
    ck = int(m.get("close_ksize", 3))
    if ck > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kern)
    mask = u8.astype(bool)

    min_area = int(m.get("min_area_px", 2))
    if min_area > 1:
        mask = _drop_small_components(mask, min_area)

    if m.get("fill_holes", True):
        mask = ndi.binary_fill_holes(mask)

    return mask.astype(bool)


def make_label_map(patch: np.ndarray, cls: str, cfg: dict, cls_index: dict[str, int]):
    """
    Build the integer label map for one image and a few diagnostics.

    Returns (label_map uint8, info dict). label_map is 0 (bg) or the class index
    on defect pixels.

    Every image here contains a defect (design_decisions.md decision §4). If the standard
    pipeline finds nothing — a genuine but tiny/scattered defect erased by the
    min-area filter — we retry keeping single pixels so the defect is still
    labeled rather than lost. The fallback is flagged in the info dict.
    """
    mask = binary_defect_mask(patch, cfg)
    fallback = False
    if mask.sum() == 0 and cfg.get("morphology", {}).get("fallback_keep_singletons", True):
        relaxed = {**cfg, "morphology": {**cfg.get("morphology", {}),
                                         "open_ksize": 0, "min_area_px": 1}}
        mask = binary_defect_mask(patch, relaxed)
        fallback = mask.sum() > 0
    label_map = np.zeros(patch.shape, dtype=np.uint8)
    label_map[mask] = cls_index[cls]

    area = int(mask.sum())
    n_comp = int(cc_label(mask).max())
    info = {
        "defect_area_px": area,
        "defect_area_frac": round(area / mask.size, 6),
        "n_components": n_comp,
        "mask_empty": area == 0,
        "used_fallback": fallback,
    }
    return label_map, info
