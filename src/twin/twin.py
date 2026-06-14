"""
twin.py — the weld Digital-Twin layer (architecture.md Part 7).

A digital twin is a virtual replica of the physical weld that reproduces its
health under inspection. Here the twin holds: the defect map (each defect placed
at its centroid in weld coordinates, from characterization), a HEALTH INDEX, and
a pass/fail status. Inspection history / RUL are optional extensions.

IMPORTANT (design_decisions.md §3.13 / §acceptance-criteria): the health index and
pass/fail rules below are a TUNABLE PLACEHOLDER, NOT validated acceptance
criteria. Calibrate the capacity and thresholds to a real standard (ISO 5817 /
ASME BPVC) with a domain expert before any real decision. Health/pass-fail is
driven by mask-derived defect presence + severity, never by a "normal" class.
"""

from __future__ import annotations


def weld_health(defects: list[dict], cfg: dict | None = None) -> dict:
    """
    Roll defect severities up into a single weld health view.

    health_index in [0,1] (1 = pristine) = 1 - min(1, Σ severity_score / capacity).
    status: FAIL if any critical defect; REVIEW if health is low or several
    moderate defects; else PASS.
    """
    cfg = cfg or {}
    capacity = float(cfg.get("capacity", 5.0))           # placeholder "budget" of severity
    review_health = float(cfg.get("review_health", 0.7))
    review_moderate = int(cfg.get("review_moderate_count", 2))

    total_sev = sum(float(d.get("severity_score", 0.0)) for d in defects)
    health = max(0.0, 1.0 - min(1.0, total_sev / capacity)) if capacity > 0 else 0.0

    n_critical = sum(1 for d in defects if d.get("severity") == "critical")
    n_moderate = sum(1 for d in defects if d.get("severity") == "moderate")
    n_minor = sum(1 for d in defects if d.get("severity") == "minor")

    if n_critical > 0:
        status = "FAIL"
    elif health < review_health or n_moderate >= review_moderate:
        status = "REVIEW"
    else:
        status = "PASS"

    return {
        "health_index": round(health, 3),
        "status": status,
        "total_severity": round(total_sev, 3),
        "n_defects": len(defects),
        "counts": {"critical": n_critical, "moderate": n_moderate, "minor": n_minor},
        "capacity": capacity,
        "note": "placeholder thresholds — calibrate to ISO 5817 / ASME with an expert",
    }


SEVERITY_COLOR = {"critical": "#d62728", "moderate": "#ff7f0e",
                  "minor": "#2ca02c", "none": "#7f7f7f"}
STATUS_COLOR = {"FAIL": "#d62728", "REVIEW": "#ff7f0e", "PASS": "#2ca02c"}
