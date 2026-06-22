"""
run_ablation.py — the data-ablation curve + plain-U-Net baseline (architecture.md §5.11).

This is the project's HEADLINE result: train the SCN-Attention U-Net AND a plain
Attention-U-Net baseline (no scattering) on shrinking fractions of the training
data, then plot accuracy vs dataset size. If the scattering curve stays flat while
the baseline collapses on little data, that proves the scattering prior's
small-data advantage — the whole reason for this architecture.

This runs many full trainings, so use a GPU (Colab). On CPU it is impractical.
Tune `--epochs` down (e.g. 60) and `--fractions` to control total time.

    # GPU (Colab): ~10 runs
    python -m scripts.run_ablation --epochs 80 --fractions 1.0 0.5 0.25 0.1

    # local CPU smoke-test of the WIRING only (tiny, meaningless accuracy)
    py -3.14 scripts/run_ablation.py --epochs 1 --fractions 1.0 0.25 --smoke
"""

from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.train import main as train_main


def run_one(fraction, scattering, epochs, config, smoke, ckpt_dir):
    # always write ablation checkpoints to a SEPARATE dir so the production
    # checkpoint (checkpoints/scn_attn_unet_best.pt) is never overwritten.
    argv = ["--config", config, "--data-fraction", str(fraction), "--epochs", str(epochs),
            "--ckpt-dir", ckpt_dir]
    if not scattering:
        argv.append("--no-scattering")
    if smoke:
        argv += ["--limit", "24"]          # tiny: only to test the wiring
    print(f"\n{'='*70}\nRUN  scattering={scattering}  fraction={fraction}  epochs={epochs}\n{'='*70}")
    return train_main(argv) or {}


def _make_ablation_config(src_config, keep_aug):
    """Derive the config used for the ablation.

    The data-efficiency ablation isolates ONE variable — the scattering prior —
    versus training-set size. The composite (porosity+slag blend) and additive-noise
    augmentation in the production config are multi-defect / noise-robustness tricks;
    they are confounds here, and at small fractions they consume scarce single-defect
    images and destabilise training. So we disable them for the ablation (base
    geometric augmentation is kept). Pass --keep-aug to use the production config as-is.
    """
    if keep_aug:
        return src_config
    import yaml
    with open(src_config) as f:
        cfg = yaml.safe_load(f)
    aug = cfg.get("augment", {})
    aug["composite_p"] = 0.0
    aug["noise_p"] = 0.0
    cfg["augment"] = aug
    derived = os.path.join("configs", "_ablation.yaml")
    with open(derived, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("ablation config: composite/noise augmentation DISABLED "
          "(isolating the scattering prior). ->", derived)
    return derived


def main():
    ap = argparse.ArgumentParser(description="Data-ablation curve (SCN vs plain U-Net)")
    ap.add_argument("--config", default="configs/model.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.1])
    ap.add_argument("--out", default="data/processed/ablation")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test wiring only")
    ap.add_argument("--keep-aug", action="store_true",
                    help="use the production augmentation as-is (default: disable "
                         "composite/noise aug, which confounds the data-efficiency test)")
    args = ap.parse_args()

    args.config = _make_ablation_config(args.config, args.keep_aug)

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join("checkpoints", "ablation")     # isolated from production ckpt
    results = {True: [], False: []}        # scattering -> list of metric dicts
    for scattering in (True, False):
        for frac in sorted(args.fractions, reverse=True):
            m = run_one(frac, scattering, args.epochs, args.config, args.smoke, ckpt_dir)
            if m:
                results[scattering].append(m)

    # write CSV
    import csv
    csv_path = os.path.join(args.out, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["use_scattering", "data_fraction", "n_train",
                    "best_val_dice_fg", "best_val_cls_acc"])
        for scat in (True, False):
            for m in results[scat]:
                w.writerow([m["use_scattering"], m["data_fraction"], m["n_train"],
                            round(m["best_val_dice_fg"], 4), round(m["best_val_cls_acc"], 4)])
    print(f"\nresults -> {csv_path}")

    # plot dice_fg vs data fraction
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for scat, label, style in [(True, "SCN-Attention U-Net", "o-"),
                               (False, "plain U-Net (baseline)", "s--")]:
        pts = sorted(results[scat], key=lambda m: m["data_fraction"])
        if not pts:
            continue
        fr = [m["data_fraction"] * 100 for m in pts]
        ax[0].plot(fr, [m["best_val_dice_fg"] for m in pts], style, label=label)
        ax[1].plot(fr, [m["best_val_cls_acc"] for m in pts], style, label=label)
    ax[0].set_title("Segmentation: val dice_fg vs training-data size")
    ax[1].set_title("Classification: val accuracy vs training-data size")
    for a in ax:
        a.set_xlabel("training data used (%)")
        a.grid(alpha=0.3)
        a.legend()
    ax[0].set_ylabel("val dice_fg")
    ax[1].set_ylabel("val cls acc")
    fig.tight_layout()
    png = os.path.join(args.out, "ablation_curve.png")
    fig.savefig(png, dpi=120)
    print(f"curve -> {png}")
    print("\nHeadline: if the SCN curve stays flatter than the baseline as data "
          "shrinks, that is the small-data advantage (architecture.md Part 2).")


if __name__ == "__main__":
    main()
