"""
Fast unit tests for Stage-1 pseudo-labels.
Run:  py -3.14 -m tests.test_pseudo_label   (or  py -3.14 -m pytest tests/ -q)
"""

import numpy as np

from src.data.pseudo_label import (
    class_index_map, compute_threshold, binary_defect_mask, make_label_map,
)

CFG = {
    "smooth": {"sigma": 0.0},
    "threshold": {"method": "otsu", "floor": 0.08, "offset": 0.0},
    "morphology": {"open_ksize": 0, "close_ksize": 0, "min_area_px": 2, "fill_holes": True},
}


def _synthetic_patch():
    """Near-black background with one bright 8x8 defect blob (like real data)."""
    p = np.zeros((64, 64), dtype=np.float32)
    p[28:36, 28:36] = 1.0
    return p


def test_class_index_map():
    assert class_index_map(["porosity", "slag"]) == {"porosity": 1, "slag": 2}


def test_threshold_has_floor():
    flat = np.zeros((32, 32), dtype=np.float32)   # otsu undefined-ish on flat
    t = compute_threshold(flat, {"method": "otsu", "floor": 0.08})
    assert t >= 0.08


def test_detects_bright_blob():
    mask = binary_defect_mask(_synthetic_patch(), CFG)
    assert mask.sum() > 0
    # the detected region should sit where the blob is
    ys, xs = np.where(mask)
    assert 25 <= ys.mean() <= 39 and 25 <= xs.mean() <= 39


def test_label_map_uses_class_index():
    cls_index = class_index_map(["porosity", "slag"])
    lm, info = make_label_map(_synthetic_patch(), "slag", CFG, cls_index)
    assert set(np.unique(lm)).issubset({0, 2})      # background or slag only
    assert lm.max() == 2
    assert not info["mask_empty"]
    assert info["defect_area_px"] > 0
    assert info["n_components"] == 1


def test_empty_on_blank():
    cls_index = class_index_map(["porosity", "slag"])
    lm, info = make_label_map(np.zeros((64, 64), np.float32), "porosity", CFG, cls_index)
    assert info["mask_empty"] and lm.max() == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
