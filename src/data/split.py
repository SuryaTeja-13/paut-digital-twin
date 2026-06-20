"""
split.py — deterministic train/val/test split (design_decisions.md decision §6, §7).

Two properties we care about:

  1. Stratified by class  — each split keeps the porosity/slag balance.
  2. Group-aware by weld   — every view of one physical weld (e.g. G100_1,
     G100_2, G100_3) stays in the SAME split. Splitting views across train and
     test would leak near-duplicate images and inflate accuracy. design_decisions.md calls
     this kind of leakage critical to avoid.

When group_by_weld is False we simply treat each image as its own group, so the
exact same code path does a plain stratified-by-image split.

Splitting happens BEFORE any augmentation; augmentation (a later module) is only
ever applied to the train split, never to val/test.
"""

from __future__ import annotations

import re
import random
from collections import defaultdict


def weld_id_from_stem(stem: str) -> str:
    """
    Derive the weld group key from a filename stem by stripping a trailing
    _<digit> view suffix.  'G100_2' -> 'G100',  'GS147' -> 'GS147'.
    """
    m = re.match(r"^(.*?)(?:_(\d+))?$", stem)
    return m.group(1)


def assign_splits(records: list[dict], ratios, seed: int,
                  stratify_by_class: bool = True, group_by_weld: bool = True) -> dict:
    """
    Assign a split label to every record.

    records : list of dicts, each with keys 'image_id', 'class', 'weld_id'.
    ratios  : (train, val, test) fractions summing to ~1.
    Returns : {image_id -> 'train'|'val'|'test'}.

    Algorithm (per class, so stratification is exact):
      - gather groups (weld_id -> [records]); if not group_by_weld, each image
        is its own group.
      - shuffle the group order with a fixed seed (reproducible).
      - walk the groups; assign each whole group to the split with the largest
        remaining deficit (target_count - current_count). Greedy-by-deficit hits
        the target ratios closely while keeping groups intact.
    """
    train_r, val_r, test_r = ratios
    split_names = ["train", "val", "test"]
    out: dict[str, str] = {}

    # group records by class so each class is split independently (stratified)
    by_class = defaultdict(list)
    for r in records:
        key = r["class"] if stratify_by_class else "_all_"
        by_class[key].append(r)

    for cls, recs in sorted(by_class.items()):
        # build groups within this class
        groups: dict[str, list] = defaultdict(list)
        for r in recs:
            gkey = r["weld_id"] if group_by_weld else r["image_id"]
            groups[gkey].append(r)

        group_keys = sorted(groups.keys())            # sort first -> determinism
        random.Random(seed).shuffle(group_keys)

        n_total = len(recs)
        targets = {
            "train": train_r * n_total,
            "val": val_r * n_total,
            "test": test_r * n_total,
        }
        counts = {s: 0 for s in split_names}

        for gkey in group_keys:
            members = groups[gkey]
            # pick the split currently furthest below its target
            deficits = {s: targets[s] - counts[s] for s in split_names}
            chosen = max(split_names, key=lambda s: deficits[s])
            for r in members:
                out[r["image_id"]] = chosen
            counts[chosen] += len(members)

    return out


def split_summary(records: list[dict], split_map: dict) -> str:
    """Human-readable counts of class x split, for the run summary."""
    table = defaultdict(lambda: defaultdict(int))
    for r in records:
        table[r["class"]][split_map[r["image_id"]]] += 1
    lines = ["  split      train    val   test  total",
             "  " + "-" * 38]
    grand = defaultdict(int)
    for cls in sorted(table):
        row = table[cls]
        tot = sum(row.values())
        lines.append(f"  {cls:<9}{row['train']:>7}{row['val']:>7}{row['test']:>7}{tot:>7}")
        for s in ("train", "val", "test"):
            grand[s] += row[s]
    gtot = sum(grand.values())
    lines.append("  " + "-" * 38)
    lines.append(f"  {'TOTAL':<9}{grand['train']:>7}{grand['val']:>7}{grand['test']:>7}{gtot:>7}")
    return "\n".join(lines)
