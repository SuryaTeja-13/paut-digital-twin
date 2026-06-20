"""
verify_math.py — "practical == mathematical" check (sir's point 9).

For every formula the project relies on (segmentation losses + the digital-twin
health index) we compute the value TWICE:
  (a) by the production code in src/, and
  (b) by an INDEPENDENT plain-numpy reimplementation written straight from the
      mathematical definition (no torch, no shared code path).
Then we assert the two agree to 1e-6. A mismatch means the code drifted from the
maths. Intermediate terms are printed so the worked arithmetic can go in the report.

Run:  py -3.14 -m scripts.verify_math
"""

from __future__ import annotations

import numpy as np
import torch

from src.models.losses import soft_dice_loss, focal_tversky_loss, focal_ce_loss
from src.twin.twin import weld_health

EPS = 1e-6
TOL = 1e-6

# ── a tiny, fully hand-computable example ────────────────────────────────
# 3 classes (0=background, 1=porosity, 2=slag), one 2×2 image.
# Column-per-pixel probabilities (each pixel's 3 class-probs sum to 1):
P_BG = [[0.1, 0.2], [0.7, 0.2]]
P_POR = [[0.6, 0.3], [0.2, 0.7]]
P_SLAG = [[0.3, 0.5], [0.1, 0.1]]
MASK = [[1, 2], [0, 1]]            # true class id per pixel


def build():
    probs = np.array([[P_BG, P_POR, P_SLAG]], dtype=np.float64)   # (1,3,2,2)
    mask = np.array([MASK], dtype=np.int64)                       # (1,2,2)
    onehot = np.zeros_like(probs)
    for c in range(3):
        onehot[0, c] = (mask[0] == c).astype(np.float64)
    return probs, mask, onehot


# ── independent numpy reimplementations (straight from the definitions) ──
def np_dice(probs, onehot, fg=(1, 2)):
    per = []
    for c in fg:
        p, t = probs[0, c], onehot[0, c]
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        per.append((2 * inter + EPS) / (denom + EPS))
    return 1.0 - float(np.mean(per)), per


def np_focal_tversky(probs, onehot, alpha=0.3, beta=0.7, gamma=1.0, fg=(1, 2)):
    per = []
    for c in fg:
        p, t = probs[0, c], onehot[0, c]
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tv = (tp + EPS) / (tp + alpha * fp + beta * fn + EPS)
        per.append((1.0 - tv) ** gamma)
    return float(np.mean(per)), per


def np_focal_ce(probs, mask, gamma=2.0):
    terms = []
    for i in range(2):
        for j in range(2):
            t = mask[0, i, j]
            pt = probs[0, t, i, j]
            ce = -np.log(pt)
            terms.append((1 - pt) ** gamma * ce)
    return float(np.mean(terms)), terms


def np_health(severities, capacity=5.0):
    total = sum(severities)
    return max(0.0, 1.0 - min(1.0, total / capacity)), total


def check(name, hand, code, extra=""):
    ok = abs(hand - code) < TOL
    flag = "OK " if ok else "MISMATCH"
    print(f"  [{flag}] {name:<22} hand={hand:.6f}  code={code:.6f}  {extra}")
    return ok


def main():
    probs, mask, onehot = build()
    tp = torch.tensor(probs)
    th = torch.tensor(onehot)
    tm = torch.tensor(mask)
    # focal_ce needs logits; log(probs) reproduces these probs through softmax
    logits = torch.log(tp.clamp_min(1e-12))

    print("=" * 70)
    print("MATH == CODE VERIFICATION  (2x2 image, 3 classes, hand-computable)")
    print("=" * 70)
    all_ok = True

    h, per = np_dice(probs, onehot)
    c = float(soft_dice_loss(tp, th, include_bg=False))
    all_ok &= check("foreground Dice loss", h, c,
                    f"(per-class Dice {[round(float(x),4) for x in per]})")

    h, per = np_focal_tversky(probs, onehot, 0.3, 0.7, 1.0)
    c = float(focal_tversky_loss(tp, th, 0.3, 0.7, 1.0, include_bg=False))
    all_ok &= check("Focal-Tversky loss", h, c,
                    f"(per-class (1-Tv) {[round(float(x),4) for x in per]})")

    h, _ = np_focal_ce(probs, mask, gamma=2.0)
    c = float(focal_ce_loss(logits, tm, gamma=2.0))
    all_ok &= check("Focal-CE loss", h, c)

    # health index -- two scenarios
    sev = [0.6, 0.6]
    h, total = np_health(sev, 5.0)
    res = weld_health([{"severity_score": s, "severity": "moderate"} for s in sev],
                      {"capacity": 5.0})
    all_ok &= check("health index (2x0.6)", h, res["health_index"],
                    f"(sum_sev={total} -> status {res['status']})")

    sev = [1.5, 2.0, 1.0]
    h, total = np_health(sev, 5.0)
    res = weld_health([{"severity_score": s, "severity": "minor"} for s in sev],
                      {"capacity": 5.0})
    all_ok &= check("health index (sum=4.5)", h, res["health_index"],
                    f"(sum_sev={total} -> status {res['status']})")

    print("-" * 70)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
