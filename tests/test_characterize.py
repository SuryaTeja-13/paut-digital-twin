"""
Unit tests for characterization. Run: py -3.14 -m tests.test_characterize
"""

import numpy as np
from src.characterize.characterize import characterize_mask, severity, summarize

CLASSES = ["porosity", "slag"]
SEV = {"type_weight": {"porosity": 1.0, "slag": 1.2}, "size_ref_px": 200.0,
       "orientation_factor": 1.0, "location_factor": 1.0,
       "cutoffs": {"moderate": 0.5, "critical": 1.0}}


def test_counts_and_type_per_instance():
    m = np.zeros((64, 64), dtype=np.int64)
    m[10:14, 10:30] = 1          # one porosity streak
    m[40:44, 40:60] = 2          # one slag streak
    d = characterize_mask(m, CLASSES, pixel_to_mm=1.0, severity_cfg=SEV)
    assert len(d) == 2
    types = {x["type"] for x in d}
    assert types == {"porosity", "slag"}


def test_orientation_and_aspect():
    m = np.zeros((64, 64), dtype=np.int64)
    m[30:32, 10:50] = 1          # long horizontal bar -> high aspect ratio
    d = characterize_mask(m, CLASSES, pixel_to_mm=1.0, severity_cfg=SEV)[0]
    assert d["aspect_ratio"] > 5
    assert d["length_mm"] > d["width_mm"]


def test_pixel_to_mm_scaling():
    m = np.zeros((64, 64), dtype=np.int64)
    m[20:30, 20:30] = 1
    d1 = characterize_mask(m, CLASSES, pixel_to_mm=1.0, severity_cfg=SEV)[0]
    d2 = characterize_mask(m, CLASSES, pixel_to_mm=2.0, severity_cfg=SEV)[0]
    assert abs(d2["length_mm"] - 2 * d1["length_mm"]) < 1e-3      # mm scales linearly
    assert abs(d2["area_mm2"] - 4 * d1["area_mm2"]) < 1e-2        # area scales with s^2
    assert abs(d2["aspect_ratio"] - d1["aspect_ratio"]) < 1e-6   # scale-free unchanged


def test_severity_levels_and_order():
    small = {"type": "porosity", "area_px": 20}
    big = {"type": "slag", "area_px": 400}
    _, s_small = severity(small, SEV)
    _, s_big = severity(big, SEV)
    assert s_small == "minor"
    assert s_big == "critical"


def test_summary_empty_and_nonempty():
    empty = summarize([], 1.0)
    assert empty["n_defects"] == 0 and empty["defect_present"] is False
    m = np.zeros((64, 64), dtype=np.int64); m[10:20, 10:40] = 2
    d = characterize_mask(m, CLASSES, severity_cfg=SEV)
    summ = summarize(d, 1.0)
    assert summ["defect_present"] and summ["dominant_type"] == "slag"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
