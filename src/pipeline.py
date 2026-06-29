"""
pipeline.py — the ONE-CLICK integration callable (architecture.md §7.4).

    image  ->  preprocess (S1)  ->  model (S2)  ->  characterization (S2)
           ->  XAI (S3)         ->  health/twin (S4)

One image in, one structured result out — everything the dashboard needs. The
model is loaded once (construct the Pipeline once, call .analyze() many times).

CLI:
    py -3.14 -m src.pipeline --image data/raw/slag/GS1.jpg
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import yaml

from .data.preprocess import process_image
from .models.infer import load_model, predict_patch, predict_batch, get_device
from .characterize.characterize import characterize_mask, summarize
from .twin.twin import weld_health


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Pipeline:
    """Load the model + configs once; analyze images on demand."""

    def __init__(self, checkpoint="checkpoints/scn_attn_unet_best.pt",
                 preprocess_cfg="configs/preprocess.yaml",
                 characterize_cfg="configs/characterize.yaml",
                 twin_cfg="configs/twin.yaml", device=None):
        self.device = device or get_device()
        self.model, self.ckpt_cfg = load_model(checkpoint, self.device)
        self.pre = _load(preprocess_cfg)
        self.char = _load(characterize_cfg)
        self.twin = _load(twin_cfg)
        self.classes = self.ckpt_cfg["data"]["classes"]
        # scattering-feature type classifier (reliable porosity-vs-slag, ~0.88
        # balanced). Optional: fall back to seg-derived type if not present.
        self.type_clf = None
        tc_path = "checkpoints/type_classifier.pkl"
        if os.path.exists(tc_path):
            from .models.type_classifier import TypeClassifier
            self.type_clf = TypeClassifier.load(tc_path)

    def _type_of(self, patch):
        """(type_index, confidence) from the scattering classifier, or (None, None)."""
        if self.type_clf is None:
            return None, None
        from .models.type_classifier import scatter_features
        feat = scatter_features(self.model.scat, patch[None].astype(np.float32), self.device)
        tidx, tprob = self.type_clf.predict_features(feat)
        ti = int(tidx[0])
        return ti, float(tprob[0][ti])

    def analyze(self, image_path: str, run_xai: bool = True, pixel_to_mm: float = None) -> dict:
        """Single image -> full result. (Mini-batch many images with analyze_batch.)"""
        patch, meta = process_image(image_path, self.pre)            # 1. preprocess
        pred = predict_patch(self.model, patch, self.device)          # 2. model
        type_idx, type_prob = self._type_of(patch)                    # 3. type
        return self._assemble(image_path, patch, meta, pred, type_idx, type_prob,
                              run_xai, pixel_to_mm)

    def analyze_batch(self, image_paths, batch_size: int = 8, run_xai: bool = False,
                      pixel_to_mm: float = None) -> list:
        """
        Mini-batch inference over many images: the model forward and the scattering
        type-features are computed in batches of `batch_size`, then each image's
        characterization + twin is assembled. XAI is off by default (per-image,
        slower) so a live feed stays responsive.
        """
        from .models.type_classifier import scatter_features
        results = []
        for start in range(0, len(image_paths), batch_size):
            chunk = image_paths[start:start + batch_size]
            patches, metas = [], []
            for p in chunk:
                patch, meta = process_image(p, self.pre)
                patches.append(patch)
                metas.append(meta)
            preds = predict_batch(self.model, patches, self.device)   # ONE forward per batch
            tidx = tprob = None
            if self.type_clf is not None:
                feats = scatter_features(
                    self.model.scat,
                    np.stack([pt.astype(np.float32) for pt in patches]), self.device)
                tidx, tprob = self.type_clf.predict_features(feats)
            for i, (path, patch, meta, pred) in enumerate(zip(chunk, patches, metas, preds)):
                if tidx is not None:
                    ti = int(tidx[i])
                    tp = float(tprob[i][ti])
                else:
                    ti, tp = None, None
                results.append(self._assemble(path, patch, meta, pred, ti, tp,
                                              run_xai, pixel_to_mm))
        return results

    def _assemble(self, image_path, patch, meta, pred, type_idx, type_prob,
                  run_xai, pixel_to_mm) -> dict:
        """Build the full result dict from a (possibly batched) prediction + type call."""
        seg, seg_prob = pred["seg"], pred["seg_prob"]

        # TYPE: scattering-feature classifier (reliable). Detection comes from the
        # segmentation; we apply the image-level type to all detected defects (every
        # image is single-type, design_decisions.md §2).
        if pixel_to_mm is None:
            pixel_to_mm = self.char.get("pixel_to_mm", self.pre.get("pixel_to_mm", 1.0))

        type_conf, type_method = None, "segmentation"
        char_mask = seg
        if type_idx is not None:
            type_conf = type_prob
            type_method = "scattering_classifier"
            # relabel ALL detected foreground to the image-level type
            char_mask = np.where(seg > 0, type_idx + 1, 0).astype(seg.dtype)

        defects = characterize_mask(char_mask, self.classes, pixel_to_mm,
                                    self.char.get("severity"),
                                    self.char.get("min_area_px", 2), seg_prob=seg_prob)
        if type_conf is not None:                          # confidence = type-call probability
            for d in defects:
                d["confidence"] = round(type_conf, 3)
        summary = summarize(defects, pixel_to_mm)

        # "no defect present" from an empty mask (design_decisions.md §3.4)
        empty_thr = self.twin.get("empty_mask_area_px", 10)
        summary["defect_present"] = int((seg > 0).sum()) >= empty_thr

        # 4. health / twin
        health = weld_health(defects, self.twin.get("health"))
        if not summary["defect_present"]:
            health["status"] = "PASS"

        result = {
            "image": os.path.basename(image_path),
            "patch": patch, "seg": seg, "seg_prob": seg_prob,
            # the exact mask characterization ran on (foreground relabeled to the
            # image-level type) — lets the dashboard outline ONE defect per card.
            "char_mask": char_mask,
            # PRIMARY type = scattering-feature classifier (~0.88 balanced, porosity ~0.86).
            "defect_type": summary["dominant_type"],
            "type_method": type_method, "type_confidence": type_conf,
            # the neural classification head's opinion, diagnostic only (weak head).
            "classifier_type": self.classes[pred["cls_idx"]],
            "classifier_prob": float(max(pred["cls_prob"])),
            "cls_idx": pred["cls_idx"], "cls_prob": pred["cls_prob"].tolist(),
            "defects": defects, "summary": summary, "health": health,
            "pixel_to_mm": pixel_to_mm, "meta": meta,
        }

        # 5. XAI: explain the channel the SEGMENTATION actually activated (where
        #    the detected pixels are), not necessarily the relabeled type.
        if run_xai and defects:
            from .xai.gradcam import grad_cam
            from .xai.faithfulness import deletion_insertion, pointing_game, trust_score
            areas = [int((seg == ci + 1).sum()) for ci in range(len(self.classes))]
            tc = int(np.argmax(areas)) + 1
            dom = result["defect_type"]
            # Grad-CAM++ — the most faithful method in the XAI comparison study.
            cam, _ = grad_cam(self.model, patch, head="seg", target_class=tc,
                              plus_plus=True, device=self.device)
            d_auc, i_auc = deletion_insertion(self.model, patch, cam, head="seg",
                                              target_class=tc, steps=15, device=self.device)
            pg = pointing_game(cam, (seg == tc).astype(int))
            from .xai.attention import attention_maps
            result["xai"] = {"cam": cam, "target_type": dom,
                             "deletion_auc": round(d_auc, 3), "insertion_auc": round(i_auc, 3),
                             "pointing_hit": None if pg != pg else float(pg),
                             "trust_score": round(trust_score(d_auc, i_auc, pg), 3),
                             "attention": attention_maps(self.model, patch, self.device)}
        return result


def _json_safe(result: dict) -> dict:
    """Drop big arrays so the result is JSON-serialisable (defects.json)."""
    keep = {k: v for k, v in result.items()
            if k not in ("patch", "seg", "seg_prob", "char_mask", "xai", "meta")}
    if "xai" in result:
        keep["xai"] = {k: v for k, v in result["xai"].items() if k not in ("cam", "attention")}
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(description="One-click PAUT pipeline")
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/scn_attn_unet_best.pt")
    ap.add_argument("--out", default=None, help="write defects.json here")
    ap.add_argument("--no-xai", action="store_true")
    args = ap.parse_args(argv)

    pipe = Pipeline(checkpoint=args.checkpoint)
    res = pipe.analyze(args.image, run_xai=not args.no_xai)
    js = _json_safe(res)
    print(json.dumps(js, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(js, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
