"""
Tests for the digital-twin health logic + pipeline result shaping.
Run: py -3.14 -m tests.test_twin
"""

from src.twin.twin import weld_health, SEVERITY_COLOR, STATUS_COLOR
from src.pipeline import _json_safe
import numpy as np


def test_health_pass_when_clean():
    h = weld_health([])
    assert h["health_index"] == 1.0 and h["status"] == "PASS"
    assert h["n_defects"] == 0


def test_health_fail_on_critical():
    defects = [{"severity": "critical", "severity_score": 2.0},
               {"severity": "minor", "severity_score": 0.1}]
    h = weld_health(defects)
    assert h["status"] == "FAIL"
    assert h["counts"]["critical"] == 1


def test_health_review_on_low_health():
    # several moderate defects -> REVIEW
    defects = [{"severity": "moderate", "severity_score": 0.6} for _ in range(3)]
    h = weld_health(defects, {"capacity": 5.0, "review_moderate_count": 2})
    assert h["status"] == "REVIEW"


def test_health_index_decreases_with_severity():
    low = weld_health([{"severity": "minor", "severity_score": 0.2}])
    high = weld_health([{"severity": "moderate", "severity_score": 2.0}])
    assert high["health_index"] < low["health_index"]


def test_color_maps_complete():
    for k in ("critical", "moderate", "minor", "none"):
        assert k in SEVERITY_COLOR
    for k in ("PASS", "REVIEW", "FAIL"):
        assert k in STATUS_COLOR


def test_json_safe_drops_arrays():
    res = {"image": "x.jpg", "patch": np.zeros((4, 4)), "seg": np.zeros((4, 4)),
           "seg_prob": np.zeros((3, 4, 4)), "defects": [], "summary": {},
           "health": {}, "xai": {"cam": np.zeros((4, 4)), "trust_score": 0.9}}
    js = _json_safe(res)
    assert "patch" not in js and "seg" not in js and "seg_prob" not in js
    assert "cam" not in js["xai"] and js["xai"]["trust_score"] == 0.9
    assert js["image"] == "x.jpg"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
