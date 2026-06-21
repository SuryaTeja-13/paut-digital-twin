"""
batch_infer.py — run the pipeline over many images with mini-batch processing (sir 6 + 7).

A clean inference entry point: point it at a folder (or glob) of TFM images and it runs
them through the pipeline in mini-batches (one model forward per batch, scattering features
batched too), then writes a per-image results table and a JSON. It also times mini-batch vs
one-at-a-time inference so the speed-up from batching is visible.

Run:
    py -3.14 -m scripts.batch_infer --dir data/raw/slag --limit 15 --batch-size 8
    py -3.14 -m scripts.batch_infer --glob "data/raw/**/*.jpg" --out data/processed/batch
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import pandas as pd

from src.pipeline import Pipeline


def gather(args):
    if args.glob:
        paths = sorted(glob.glob(args.glob, recursive=True))
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.dir, e)))
    return paths[:args.limit] if args.limit else paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Mini-batch inference over a folder of images")
    ap.add_argument("--dir", default="data/raw/slag", help="folder of images")
    ap.add_argument("--glob", default=None, help="glob pattern (overrides --dir)")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="data/processed/batch")
    ap.add_argument("--time-compare", action="store_true",
                    help="also time one-at-a-time inference to show the batch speed-up")
    args = ap.parse_args(argv)

    paths = gather(args)
    if not paths:
        print("no images found")
        return 1
    os.makedirs(args.out, exist_ok=True)
    pipe = Pipeline()
    print(f"loaded pipeline on {pipe.device}; {len(paths)} images, batch size {args.batch_size}")

    t0 = time.perf_counter()
    results = pipe.analyze_batch(paths, batch_size=args.batch_size, run_xai=False)
    batch_t = time.perf_counter() - t0

    rows = []
    for r in results:
        rows.append({"image": r["image"], "defect_type": r["defect_type"],
                     "type_confidence": r["type_confidence"],
                     "n_defects": r["health"]["n_defects"],
                     "health_index": r["health"]["health_index"],
                     "status": r["health"]["status"],
                     "defect_present": r["summary"]["defect_present"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "batch_results.csv"), index=False)
    with open(os.path.join(args.out, "batch_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{df.to_string(index=False)}")
    print(f"\nmini-batch inference: {len(paths)} images in {batch_t:.2f}s "
          f"({batch_t / len(paths) * 1000:.0f} ms/image)")
    print(f"  PASS {int((df['status'] == 'PASS').sum())} | "
          f"REVIEW {int((df['status'] == 'REVIEW').sum())} | "
          f"FAIL {int((df['status'] == 'FAIL').sum())}")

    if args.time_compare:
        t0 = time.perf_counter()
        for p in paths:
            pipe.analyze(p, run_xai=False)
        seq_t = time.perf_counter() - t0
        speedup = seq_t / batch_t if batch_t > 0 else float("nan")
        print(f"\none-at-a-time: {seq_t:.2f}s  |  mini-batch: {batch_t:.2f}s  "
              f"-> {speedup:.2f}x faster")

    print(f"\nwrote {args.out}/batch_results.csv (+ .json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
