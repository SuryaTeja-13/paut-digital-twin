"""
Fast unit tests for Milestone 1 — run with:  py -3.14 -m pytest tests/ -q
(or plain  py -3.14 tests/test_preprocess.py  for a no-pytest run)

These use tiny synthetic arrays so they run instantly on CPU and assert the
properties design_decisions.md cares about: single channel, [0,1] range, frame removal,
and — most importantly — no weld leakage across splits.
"""

import numpy as np

from src.data.preprocess import (
    detect_source_type, to_amplitude, crop_border, normalize, resize_patch,
)
from src.data.split import assign_splits, weld_id_from_stem


def test_grayscale_detection_and_single_channel():
    gray = np.tile(np.arange(256, dtype=np.uint8).reshape(1, 256, 1), (256, 1, 3))
    assert detect_source_type(gray) == "gray"
    amp = to_amplitude(gray, {"color_method": "luminance"}, "gray")
    assert amp.ndim == 2                       # exactly one amplitude channel


def test_color_detection():
    color = np.zeros((32, 32, 3), dtype=np.uint8)
    color[..., 0] = 200      # strong red only -> colorful
    assert detect_source_type(color) == "color"


def test_crop_removes_white_frame():
    img = np.full((100, 100), 255.0, dtype=np.float32)   # all-white frame
    img[20:80, 30:70] = 10.0                              # dark data region
    cropped, bbox = crop_border(img, {"enabled": True, "white_thresh": 250,
                                       "white_line_frac": 0.98, "margin": 0})
    r0, r1, c0, c1 = bbox
    assert 18 <= r0 <= 22 and 78 <= r1 <= 82
    assert 28 <= c0 <= 32 and 68 <= c1 <= 72
    assert cropped.max() < 250                            # frame is gone


def test_crop_trims_antialiased_rim():
    # white(255) frame, then a gray rim row(175) just inside it, then dark data.
    img = np.full((100, 100), 255.0, dtype=np.float32)
    img[20:80, 20:80] = 10.0          # dark data
    img[20, 20:80] = 175.0            # bright gray rim on the top data row
    img[79, 20:80] = 175.0            # and the bottom data row
    cropped, _ = crop_border(img, {"enabled": True, "white_thresh": 250,
                                    "white_line_frac": 0.98, "margin": 0,
                                    "trim_bright_band": True,
                                    "band_bright_thresh": 40, "band_margin": 0})
    # the 175 rim must be gone from the cropped edges
    assert cropped[0].max() < 40 and cropped[-1].max() < 40


def test_crop_keeps_localized_edge_defect():
    # a localized bright defect touching an edge must NOT be trimmed away
    img = np.full((100, 100), 255.0, dtype=np.float32)
    img[20:80, 20:80] = 10.0
    img[20:25, 45:50] = 255.0         # small bright blob at the top edge of data
    cropped, _ = crop_border(img, {"enabled": True, "white_thresh": 250,
                                    "white_line_frac": 0.98, "margin": 0,
                                    "trim_bright_band": True,
                                    "band_bright_thresh": 40, "band_margin": 0})
    assert cropped.max() > 200          # the blob survived


def test_normalize_minmax_range():
    a = np.array([[0.0, 50.0], [100.0, 200.0]], dtype=np.float32)
    out = normalize(a, {"method": "minmax"})
    assert abs(out.min()) < 1e-6 and abs(out.max() - 1.0) < 1e-6


def test_resize_shape():
    a = np.random.rand(420, 398).astype(np.float32)
    out = resize_patch(a, 256, {"mode": "stretch", "interpolation": "area"})
    assert out.shape == (256, 256)


def test_weld_id_parsing():
    assert weld_id_from_stem("G100_2") == "G100"
    assert weld_id_from_stem("GS147") == "GS147"


def test_split_ratios_and_no_group_leakage():
    # 30 welds per class, 3 views each -> 180 images
    records = []
    for cls in ("porosity", "slag"):
        for w in range(30):
            for v in range(1, 4):
                records.append({"image_id": f"{cls}__W{w}_{v}",
                                "class": cls, "weld_id": f"{cls}__W{w}"})
    sm = assign_splits(records, (0.70, 0.15, 0.15), seed=42,
                       stratify_by_class=True, group_by_weld=True)

    # no weld in more than one split
    weld_splits = {}
    for r in records:
        weld_splits.setdefault(r["weld_id"], set()).add(sm[r["image_id"]])
    assert all(len(s) == 1 for s in weld_splits.values()), "weld leaked across splits!"

    # ratios roughly honored per class
    for cls in ("porosity", "slag"):
        n = sum(1 for r in records if r["class"] == cls)
        tr = sum(1 for r in records if r["class"] == cls and sm[r["image_id"]] == "train")
        assert 0.6 <= tr / n <= 0.8

    # determinism: same seed -> same split
    sm2 = assign_splits(records, (0.70, 0.15, 0.15), seed=42,
                        stratify_by_class=True, group_by_weld=True)
    assert sm == sm2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
