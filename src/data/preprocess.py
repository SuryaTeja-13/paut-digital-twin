"""
preprocess.py — turn a raw TFM image into one normalized amplitude patch.

Pipeline for a single image (design_decisions.md decision §1, architecture.md Part 4):

    load RGB  ->  to single amplitude channel  ->  crop white frame
              ->  (optional) gentle denoise     ->  resize to 256x256
              ->  per-image normalize           ->  float32 patch in [0,1]

WHY a single amplitude channel: a TFM image is a scalar amplitude field; color
is only a colormap painted on top and carries no extra physics. Collapsing to
one channel makes the model work identically on color OR grayscale inputs.

These are pure functions (no disk I/O) so they are easy to unit-test. The disk
side (scanning folders, saving .npy, manifest) lives in build_dataset.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import cv2


# ───────────────────────── load + amplitude ──────────────────────────────

def load_rgb(path: str) -> np.ndarray:
    """Load any image as an HxWx3 uint8 RGB array (grayscale is expanded to 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def detect_source_type(rgb: np.ndarray, gray_tolerance: int = 6) -> str:
    """
    Decide whether an image is genuinely grayscale or carries color.

    Many "grayscale" exports are stored as 3 identical RGB channels. We call an
    image grayscale if, for almost every pixel, the spread between its max and
    min channel is tiny.
    """
    spread = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    frac_colorful = float((spread > gray_tolerance).mean())
    return "gray" if frac_colorful < 0.01 else "color"


def _colormap_lut(name: str, n: int = 256) -> np.ndarray:
    """Return an (n, 3) float LUT in [0,255] for a matplotlib colormap."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(name, n)
    lut = (np.asarray([cmap(i)[:3] for i in range(n)]) * 255.0)
    return lut.astype(np.float32)


def colormap_inverse(rgb: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """
    FUTURE color path: invert a known colormap back to scalar amplitude.

    For each pixel, find the nearest entry in the colormap's lookup table; that
    entry's index (0..255) is the recovered amplitude. Done in row chunks to
    keep memory bounded. Not used for the current grayscale data, but wired in
    so color TFM images need NO code change later (design_decisions.md decision §1).
    """
    lut = _colormap_lut(colormap)                  # (256, 3)
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(np.float32)
    amp = np.empty(flat.shape[0], dtype=np.float32)
    chunk = 20000
    for s in range(0, flat.shape[0], chunk):
        block = flat[s:s + chunk]                  # (c, 3)
        d = ((block[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)   # (c, 256)
        amp[s:s + chunk] = np.argmin(d, axis=1)
    return amp.reshape(h, w)                        # 0..255 amplitude


def to_amplitude(rgb: np.ndarray, amp_cfg: dict, source_type: str) -> np.ndarray:
    """
    Collapse an RGB image to one float32 amplitude channel in [0,255].

    grayscale -> luminance (channels are equal, so this is lossless).
    color     -> colormap_inverse (recover the scalar) or luminance fallback.
    """
    if source_type == "gray":
        return rgb.astype(np.float32).mean(axis=2)
    method = amp_cfg.get("color_method", "luminance")
    if method == "colormap_inverse":
        return colormap_inverse(rgb, amp_cfg.get("colormap", "jet"))
    # luminance (Rec. 601) — robust default when the colormap is unknown
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


# ───────────────────────── crop / denoise / resize ───────────────────────

def _trim_bright_band(amp, r0, r1, c0, c1, bright_thresh, margin):
    """
    Stage 2 of cropping: from each edge of the [r0:r1, c0:c1] box, advance inward
    while a FULL edge row/column is bright (the anti-aliased gray rim the white
    crop leaves behind). Stops at the first dark row/col, so localized bright
    defects near an edge are preserved — only contiguous frame bands are trimmed.
    """
    while r0 < r1 and amp[r0, c0:c1 + 1].mean() > bright_thresh:
        r0 += 1
    while r1 > r0 and amp[r1, c0:c1 + 1].mean() > bright_thresh:
        r1 -= 1
    while c0 < c1 and amp[r0:r1 + 1, c0].mean() > bright_thresh:
        c0 += 1
    while c1 > c0 and amp[r0:r1 + 1, c1].mean() > bright_thresh:
        c1 -= 1
    return r0 + margin, r1 - margin, c0 + margin, c1 - margin


def crop_border(amp: np.ndarray, crop_cfg: dict):
    """
    Remove the matplotlib frame/axes, keep the genuine TFM data region.

    Stage 1 — drop the near-white frame: a row/column that is almost entirely
              near-white is treated as frame.
    Stage 2 — trim the leftover gray rim: the frame edge is anti-aliased, so a
              1-2px bright band can survive stage 1. We trim contiguous bright
              edge bands within the box (optional, on by default).

    Returns the cropped amplitude and the bounding box (r0, r1, c0, c1) in
    original-image coordinates.
    """
    h, w = amp.shape
    if not crop_cfg.get("enabled", True):
        return amp, (0, h - 1, 0, w - 1)

    white_thresh = crop_cfg.get("white_thresh", 250)
    line_frac = crop_cfg.get("white_line_frac", 0.98)
    margin = int(crop_cfg.get("margin", 2))

    near_white = amp >= white_thresh
    rows = np.where(near_white.mean(axis=1) < line_frac)[0]
    cols = np.where(near_white.mean(axis=0) < line_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return amp, (0, h - 1, 0, w - 1)   # all frame — keep it all rather than crash

    r0, r1 = int(rows[0]) + margin, int(rows[-1]) - margin
    c0, c1 = int(cols[0]) + margin, int(cols[-1]) - margin
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h - 1, r1), min(w - 1, c1)
    if r1 <= r0 or c1 <= c0:               # margin ate everything; back off
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])

    if crop_cfg.get("trim_bright_band", True):
        br = crop_cfg.get("band_bright_thresh", 40)
        bm = int(crop_cfg.get("band_margin", 1))
        nr0, nr1, nc0, nc1 = _trim_bright_band(amp, r0, r1, c0, c1, br, bm)
        if nr1 > nr0 and nc1 > nc0:        # only apply if a valid box remains
            r0, r1, c0, c1 = nr0, nr1, nc0, nc1

    return amp[r0:r1 + 1, c0:c1 + 1], (r0, r1, c0, c1)


def denoise(amp: np.ndarray, dn_cfg: dict) -> np.ndarray:
    """Optional gentle denoise. Default 'none' to protect thin/faint defects."""
    method = dn_cfg.get("method", "none")
    if method == "none":
        return amp
    if method == "median":
        k = int(dn_cfg.get("median_ksize", 3))
        k = k if k % 2 == 1 else k + 1              # cv2 needs odd kernel
        return cv2.medianBlur(amp.astype(np.float32), k)
    if method == "nlmeans":
        u8 = cv2.normalize(amp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out = cv2.fastNlMeansDenoising(u8, None, h=7, templateWindowSize=7, searchWindowSize=21)
        return out.astype(np.float32)
    raise ValueError(f"unknown denoise method: {method}")


def resize_patch(amp: np.ndarray, size: int, resize_cfg: dict) -> np.ndarray:
    """Resize the cropped region to size x size (stretch or aspect-preserving pad)."""
    interp = {
        "area": cv2.INTER_AREA,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }[resize_cfg.get("interpolation", "area")]
    mode = resize_cfg.get("mode", "stretch")

    if mode == "pad":
        h, w = amp.shape
        side = max(h, w)
        canvas = np.full((side, side), float(amp.min()), dtype=np.float32)  # pad w/ background
        top, left = (side - h) // 2, (side - w) // 2
        canvas[top:top + h, left:left + w] = amp
        amp = canvas
    return cv2.resize(amp.astype(np.float32), (size, size), interpolation=interp)


def normalize(amp: np.ndarray, norm_cfg: dict) -> np.ndarray:
    """Per-image normalization -> float32. minmax->[0,1]; zscore->mean0/std1 (clipped)."""
    method = norm_cfg.get("method", "minmax")
    a = amp.astype(np.float32)
    if method == "minmax":
        lo, hi = float(a.min()), float(a.max())
        if hi - lo < 1e-6:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)
    if method == "zscore":
        mu, sd = float(a.mean()), float(a.std())
        if sd < 1e-6:
            return np.zeros_like(a)
        z = (a - mu) / sd
        c = float(norm_cfg.get("zscore_clip", 3.0))
        return np.clip(z, -c, c)
    raise ValueError(f"unknown normalize method: {method}")


# ───────────────────────── one-image driver ──────────────────────────────

def process_image(path: str, cfg: dict):
    """
    Run the full single-image pipeline. Returns (patch, meta).

    patch : float32 (patch_size, patch_size), normalized
    meta  : dict with source_type, colormap, crop box, original size
    """
    rgb = load_rgb(path)
    orig_h, orig_w = rgb.shape[:2]

    source_type = detect_source_type(rgb, cfg["amplitude"].get("gray_tolerance", 6))
    amp = to_amplitude(rgb, cfg["amplitude"], source_type)
    amp, bbox = crop_border(amp, cfg["crop"])
    amp = denoise(amp, cfg["denoise"])
    amp = resize_patch(amp, cfg["patch_size"], cfg["resize"])
    patch = normalize(amp, cfg["normalize"]).astype(np.float32)

    colormap = cfg["amplitude"].get("colormap", "jet") if source_type == "color" else "none"
    meta = {
        "source_type": source_type,
        "colormap": colormap,
        "orig_h": orig_h, "orig_w": orig_w,
        "crop_r0": bbox[0], "crop_r1": bbox[1],
        "crop_c0": bbox[2], "crop_c1": bbox[3],
    }
    return patch, meta
