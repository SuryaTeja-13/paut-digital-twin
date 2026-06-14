"""
check_contamination.py — guard against mislabeled / duplicate images.

After the earlier incident (256 slag images mislabeled as porosity), this is a
standard pre-training check. It reports:
  - per-class counts and filename-prefix breakdown
  - byte-identical images ACROSS the porosity/slag folders (the contamination)
  - duplicate images WITHIN each folder
  - overlap with the quarantine folder (did the old bad files come back?)

Run from the project root:
    py -3.14 scripts/check_contamination.py
"""

from __future__ import annotations

import os
import sys
import glob
import hashlib
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def files(d):
    return sorted(glob.glob(os.path.join(ROOT, d, "*.jpg")) +
                  glob.glob(os.path.join(ROOT, d, "*.jpeg")) +
                  glob.glob(os.path.join(ROOT, d, "*.png")))


def prefix(name):
    if name.startswith("GS"):
        return "GS#"
    if name.startswith("G"):
        return "G#"
    return "other"


def main():
    por = files("data/raw/porosity")
    slag = files("data/raw/slag")
    print(f"porosity: {len(por)} files   slag: {len(slag)} files\n")

    por_names = [os.path.basename(f) for f in por]
    slag_names = [os.path.basename(f) for f in slag]
    print("porosity name prefixes:", dict(Counter(prefix(n) for n in por_names)))
    print("slag name prefixes:    ", dict(Counter(prefix(n) for n in slag_names)), "\n")

    por_hash = {f: md5(f) for f in por}
    slag_hash = {f: md5(f) for f in slag}
    por_h2n = defaultdict(list)
    for f, h in por_hash.items():
        por_h2n[h].append(os.path.basename(f))
    slag_set = set(slag_hash.values())

    # cross-folder byte-identical
    cross = [(n, h) for f, h in por_hash.items() for n in [os.path.basename(f)] if h in slag_set]
    print(f"[1] byte-IDENTICAL across porosity<->slag: {len(cross)}  "
          + ("**CONTAMINATION**" if cross else "(clean)"))
    for n, _ in cross[:10]:
        print("     ", n)

    # within-folder duplicates
    for cls, hashes in [("porosity", por_hash), ("slag", slag_hash)]:
        h2 = defaultdict(list)
        for f, h in hashes.items():
            h2[h].append(os.path.basename(f))
        dups = {h: v for h, v in h2.items() if len(v) > 1}
        print(f"[2] duplicate images within {cls}: {len(dups)} groups "
              f"({sum(len(v) for v in dups.values())} files)")
        for v in list(dups.values())[:5]:
            print("     ", v)

    # overlap with quarantine (did the old mislabeled files come back?)
    q = files("data/quarantine/porosity_mislabeled_GS")
    if q:
        q_hashes = {md5(f) for f in q}
        back = [os.path.basename(f) for f, h in por_hash.items() if h in q_hashes]
        print(f"[3] new porosity files identical to QUARANTINED slag: {len(back)}  "
              + ("**BAD — these are the old mislabeled files**" if back else "(clean)"))
        for n in back[:10]:
            print("     ", n)

    ok = not cross and (not q or not back)
    print("\nRESULT:", "OK — safe to train" if ok else "PROBLEM — do NOT train until fixed")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
