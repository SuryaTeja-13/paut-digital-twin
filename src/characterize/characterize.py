"""
characterize.py — measure each defect from its mask (architecture.md §5.7, design_decisions.md §11).

Per design_decisions.md decision §11, defect properties are DERIVED FROM THE MASK, not
regressed. For every defect instance (a connected component in a class channel)
we compute size, shape, orientation, location and a severity score.

Units: pixel_to_mm defaults to 1.0 (unknown scale) -> mm values equal pixel
values for now. Pass the real mm/pixel later and the *_mm fields become correct
with NO code change. Scale-free fields (orientation, aspect ratio, eccentricity)
are already correct.

Defect TYPE is taken from the segmentation channel the instance lives in
(seg-derived type — empirically more reliable than the global-pooled classifier).

Severity = f(type, size, orientation, location) is a TUNABLE PLACEHOLDER scaffold.
Do NOT read the cut-offs as acceptance criteria — calibrate to ISO 5817 / ASME
BPVC with a domain expert before any pass/fail use (design_decisions.md §11).
"""

from __future__ import annotations

import math
from skimage.measure import label, regionprops


def _instances(label_map, class_index):
    """Connected components within one class channel -> regionprops list."""
    binary = (label_map == class_index)
    if not binary.any():
        return []
    return regionprops(label(binary))


def characterize_mask(label_map, classes, pixel_to_mm=1.0, severity_cfg=None,
                      min_area_px=2, seg_prob=None):
    """
    label_map : (H,W) ints — 0=background, 1=classes[0], 2=classes[1], ...
    classes   : ["porosity", "slag"]  (class index = position+1)
    seg_prob  : optional (C,H,W) softmax probabilities -> per-defect confidence.

    Returns a list of per-defect dicts (the defects.json records).
    """
    s = float(pixel_to_mm)
    defects = []
    for ci, cname in enumerate(classes):
        cls_index = ci + 1
        for region in _instances(label_map, cls_index):
            if region.area < min_area_px:
                continue
            major = float(region.axis_major_length)
            minor = float(region.axis_minor_length)
            aspect = major / minor if minor > 1e-6 else float(major)
            cy, cx = region.centroid                       # (row, col)
            # skimage orientation: radians from the row-axis to the major axis
            orient_deg = float(math.degrees(region.orientation))

            conf = None
            if seg_prob is not None:
                m = region.image                            # local bool mask
                rr0, cc0, rr1, cc1 = region.bbox
                conf = float(seg_prob[cls_index, rr0:rr1, cc0:cc1][m].mean())

            d = {
                "type": cname,
                "area_px": int(region.area),
                "area_mm2": round(region.area * s * s, 4),
                "length_mm": round(major * s, 4),
                "width_mm": round(minor * s, 4),
                "equiv_diameter_mm": round(float(region.equivalent_diameter_area) * s, 4),
                "orientation_deg": round(orient_deg, 2),
                "aspect_ratio": round(aspect, 3),
                "eccentricity": round(float(region.eccentricity), 3),
                "solidity": round(float(region.solidity), 3),
                "centroid_px": [round(cx, 1), round(cy, 1)],     # (x, y)
                "centroid_mm": [round(cx * s, 3), round(cy * s, 3)],
                "bbox_px": [int(v) for v in region.bbox],        # (r0,c0,r1,c1)
                "confidence": round(conf, 3) if conf is not None else None,
            }
            d["severity_score"], d["severity"] = severity(d, severity_cfg)
            defects.append(d)

    # stable id per defect, largest first
    defects.sort(key=lambda x: x["area_px"], reverse=True)
    for i, d in enumerate(defects):
        d["id"] = i + 1
    return defects


def severity(defect, cfg=None):
    """
    severity = w_type * norm(size) * orientation_factor * location_factor,
    mapped to {minor, moderate, critical}.

    PLACEHOLDER scaffold (design_decisions.md §11): all weights/cut-offs are tunable and are
    NOT validated acceptance thresholds. Calibrate to ISO 5817 / ASME with a
    domain expert before any real pass/fail decision.
    """
    cfg = cfg or {}
    w_type = cfg.get("type_weight", {"porosity": 1.0, "slag": 1.2})
    size_ref = float(cfg.get("size_ref_px", 200.0))         # normalizing area
    cuts = cfg.get("cutoffs", {"moderate": 0.5, "critical": 1.0})

    wt = float(w_type.get(defect["type"], 1.0))
    norm_size = min(defect["area_px"] / size_ref, 3.0)      # cap runaway
    orientation_factor = float(cfg.get("orientation_factor", 1.0))   # placeholder
    location_factor = float(cfg.get("location_factor", 1.0))         # placeholder (no weld geometry yet)

    score = wt * norm_size * orientation_factor * location_factor
    if score >= cuts["critical"]:
        level = "critical"
    elif score >= cuts["moderate"]:
        level = "moderate"
    else:
        level = "minor"
    return round(float(score), 4), level


def summarize(defects, pixel_to_mm=1.0):
    """Image-level rollup for the digital twin: counts, dominant type, total area."""
    if not defects:
        return {"n_defects": 0, "dominant_type": None, "total_area_mm2": 0.0,
                "max_severity": "none", "defect_present": False}
    order = {"minor": 0, "moderate": 1, "critical": 2}
    by_type = {}
    for d in defects:
        by_type[d["type"]] = by_type.get(d["type"], 0) + d["area_px"]
    dominant = max(by_type, key=by_type.get)
    return {
        "n_defects": len(defects),
        "dominant_type": dominant,
        "total_area_mm2": round(sum(d["area_mm2"] for d in defects), 4),
        "max_severity": max((d["severity"] for d in defects), key=lambda s: order[s]),
        "defect_present": True,
    }
