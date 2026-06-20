"""
app.py — the Digital-Twin dashboard (architecture.md Part 7, Student 4).

Upload a TFM image (color OR grayscale) -> the one-click pipeline runs
(preprocess -> model -> characterize -> XAI -> health) -> the twin shows results
across tabs: Overview/Health, Explainability, Defects, and Details.

Defect TYPE shown is from the scattering classifier (the reliable signal, ~0.88 balanced).
The neural classification head's opinion is shown only as a labelled diagnostic.

Run from the project root:
    py -3.14 -m streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys
import glob
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from src.pipeline import Pipeline, _json_safe
from src.twin.twin import SEVERITY_COLOR, STATUS_COLOR

st.set_page_config(page_title="PAUT Weld Digital Twin", layout="wide", page_icon="🔬")


@st.cache_resource
def get_pipeline():
    return Pipeline()


# ─────────────── figure helpers ───────────────
def _square(figsize=(5, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    return fig, ax


def weld_map_fig(patch, seg, defects):
    fig, ax = _square()
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax.contour(seg > 0, levels=[0.5], colors="cyan", linewidths=0.6)
    for d in defects:
        x, y = d["centroid_px"]
        ax.scatter([x], [y], s=130, edgecolors="white",
                   c=SEVERITY_COLOR.get(d["severity"], "#fff"), zorder=3)
        ax.text(x + 5, y, str(d["id"]), color="white", fontsize=9, weight="bold")
    ax.set_title("Weld map — defects by severity")
    return fig


def heat_fig(patch, heat, title):
    fig, ax = _square()
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax.imshow(heat, cmap="jet", alpha=0.5)
    ax.set_title(title)
    return fig


def _instance_mask(inst_src, d):
    """Boolean mask of ONLY this defect's connected component.

    Cards crop a padded box around a defect, so a neighbouring defect can fall in
    the same window. We isolate the one this card describes by matching its stored
    bounding box (exact, unique per component); centroid-hit is the fallback.
    """
    from skimage.measure import label, regionprops
    lbl = label(inst_src > 0)
    target = tuple(d["bbox_px"])
    for region in regionprops(lbl):
        if tuple(region.bbox) == target:
            return lbl == region.label
    cx, cy = d["centroid_px"]                       # (col, row)
    ry = min(int(round(cy)), lbl.shape[0] - 1)
    rx = min(int(round(cx)), lbl.shape[1] - 1)
    cid = lbl[ry, rx]
    return (lbl == cid) if cid > 0 else (inst_src > 0)


def defect_crop_fig(patch, inst_src, d):
    r0, c0, r1, c1 = d["bbox_px"]
    pad = 10
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(patch.shape[0], r1 + pad), min(patch.shape[1], c1 + pad)
    this = _instance_mask(inst_src, d)
    others = (inst_src > 0) & ~this                 # neighbouring defects, if any
    fig, ax = _square((2.4, 2.4))
    ax.imshow(patch[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1)
    if others[r0:r1, c0:c1].any():                  # context: faint dashed cyan
        ax.contour(others[r0:r1, c0:c1], levels=[0.5], colors="#33ddff",
                   linewidths=0.6, linestyles="dashed")
    ax.contour(this[r0:r1, c0:c1], levels=[0.5], colors="red", linewidths=1.2)
    ax.set_title(f"#{d['id']}", color="red", fontsize=11, pad=2)
    return fig


def status_banner(h):
    color = STATUS_COLOR.get(h["status"], "#555")
    st.markdown(
        f"<div style='background:{color};padding:14px 20px;border-radius:10px;"
        f"color:white;font-size:26px;font-weight:700;text-align:center'>"
        f"WELD STATUS: {h['status']} &nbsp;·&nbsp; health {h['health_index']:.2f}</div>",
        unsafe_allow_html=True)


# ─────────────── header + input ───────────────
st.title("🔬 Explainable-AI Digital Twin — PAUT Weld Inspection")
st.caption("Detect & classify weld defects, measure them, explain the decision, and report weld health.")

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload a TFM image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    st.markdown("— or pick a sample —")
    samples = sorted(glob.glob("data/raw/porosity/*.jpg"))[:5] + \
        sorted(glob.glob("data/raw/slag/*.jpg"))[:5]
    sample = st.selectbox("Sample image", ["(none)"] + [os.path.relpath(s) for s in samples])
    run_xai = st.checkbox("Run explainability", value=True)
    st.markdown("---")
    pixel_to_mm = st.number_input("pixel → mm scale", value=1.0, min_value=0.0001,
                                  step=0.05, format="%.4f",
                                  help="Unknown for now → 1.0 (values shown in pixels). "
                                       "Set the real probe scale to get true mm.")
    unit = "mm" if abs(pixel_to_mm - 1.0) > 1e-9 else "px"

img_path = None
if up is not None:
    tmp = os.path.join(tempfile.gettempdir(), up.name)
    with open(tmp, "wb") as f:
        f.write(up.getbuffer())
    img_path = tmp
elif sample != "(none)":
    img_path = sample

if img_path is None:
    st.info("⬅️ Upload an image or choose a sample to run the pipeline.")
    st.stop()

with st.spinner("Running pipeline: preprocess → model → characterize → XAI → health…"):
    res = get_pipeline().analyze(img_path, run_xai=run_xai, pixel_to_mm=pixel_to_mm)

h, summ, defects = res["health"], res["summary"], res["defects"]
status_banner(h)
st.write("")

tab_overview, tab_xai, tab_defects, tab_details = st.tabs(
    ["🩺 Overview", "🧠 Explainability", "📋 Defects", "ℹ️ Details"])

# ── Overview ──
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health index", f"{h['health_index']:.2f}", help="1.0 = pristine")
    c2.metric("Defect type", res["defect_type"] or "—", help="scattering classifier (~0.88 balanced, reliable)")
    c3.metric("Defects found", h["n_defects"])
    c4.metric("Max severity", summ["max_severity"])
    st.caption(f"severity counts — critical {h['counts']['critical']}, "
               f"moderate {h['counts']['moderate']}, minor {h['counts']['minor']}  ·  "
               f"⚠️ thresholds are placeholders (calibrate to ISO 5817 / ASME).")
    left, right = st.columns(2)
    left.pyplot(weld_map_fig(res["patch"], res["seg"], defects))
    right.markdown("**Original (input)**")
    right.image(img_path, use_container_width=True)

# ── Explainability ──
with tab_xai:
    if run_xai and "xai" in res:
        x = res["xai"]
        c1, c2 = st.columns(2)
        c1.pyplot(heat_fig(res["patch"], x["cam"], f"Seg-Grad-CAM ({x['target_type']})"))
        c1.success(f"**Trust score {x['trust_score']:.2f}** — deletion {x['deletion_auc']:.2f} "
                   f"(↓ better) · insertion {x['insertion_auc']:.2f} (↑ better) · "
                   f"peak-on-defect {x['pointing_hit']}")
        att = x.get("attention", {})
        if att:
            key = "gate_up1" if "gate_up1" in att else list(att)[0]
            c2.pyplot(heat_fig(res["patch"], att[key], f"Model's intrinsic attention ({key})"))
            c2.caption("The model's own attention gate — free, faithful-by-construction explanation.")
        st.caption("Grad-CAM = gradient-based saliency on the decoder; trust = deletion/insertion + "
                   "pointing-game (architecture.md Part 6).")
    else:
        st.info("Enable 'Run explainability' in the sidebar to see Grad-CAM + attention.")

# ── Defects ──
with tab_defects:
    st.subheader(f"Defects ({len(defects)})")
    if not defects:
        st.success("No defect detected (mask essentially empty).")
    else:
        table = pd.DataFrame([{
            "id": d["id"], "type": d["type"], "severity": d["severity"],
            f"length_{unit}": d["length_mm"], f"width_{unit}": d["width_mm"],
            "area_px": d["area_px"], "orient_deg": d["orientation_deg"],
            "aspect": d["aspect_ratio"], "confidence": d["confidence"],
        } for d in defects])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download defects.json",
                           json.dumps(_json_safe(res), indent=2),
                           file_name=f"{os.path.splitext(res['image'])[0]}_defects.json")
        st.markdown("**Defect cards**")
        st.caption("Each card outlines **one** defect in **solid red** — the one whose numbers are "
                   "shown below it. A **faint dashed cyan** outline is a *neighbouring* defect that "
                   "happens to sit in the same crop (it has its own card). Match a card to the "
                   "weld map above by its **#id**.")
        inst_src = res.get("char_mask", res["seg"])
        cols = st.columns(min(4, len(defects)))
        for i, d in enumerate(defects):
            with cols[i % len(cols)]:
                st.pyplot(defect_crop_fig(res["patch"], inst_src, d))
                st.markdown(
                    f"**#{d['id']} · {d['type']}** "
                    f"<span style='color:{SEVERITY_COLOR[d['severity']]}'>●</span> {d['severity']}<br>"
                    f"len {d['length_mm']} · wid {d['width_mm']} {unit}<br>"
                    f"area {d['area_px']} px · {d['orientation_deg']}°",
                    unsafe_allow_html=True)

# ── Details ──
with tab_details:
    # ---- Model accuracy (test set) — NOT this-weld health ----
    st.markdown("### 📊 Model accuracy (measured on the held-out test set)")
    st.caption("This is how often the MODEL is right across many images — completely different "
               "from the weld-health number above (which scores this one weld's condition).")
    mpath = "data/processed/model_metrics.json"
    if os.path.exists(mpath):
        with open(mpath) as f:
            m = json.load(f)
        # prefer the scattering classifier (what the app uses); fall back to seg
        type_block = m.get("type_via_scattering_clf") or m["type_via_segmentation"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Detection (mean Dice)", f"{m['segmentation']['mean_fg_dice']:.2f}",
                  help="overlap of predicted vs labelled defect pixels (1.0 = perfect)")
        a2.metric("Type accuracy (balanced)", f"{type_block['balanced_acc']:.2f}",
                  help="porosity vs slag via the scattering classifier (0.5 = chance)")
        a3.metric("Missed-defect rate", f"{m['no_defect_rate']*100:.0f}%",
                  help="images wrongly called 'no defect'")
        pc = type_block["per_class"]
        st.write("- **Per-class type accuracy (scattering classifier):** "
                 + ", ".join(f"{k} {v:.2f}" for k, v in pc.items()) + ".")
        st.write("- **Per-class Dice:** "
                 + ", ".join(f"{k} {v:.2f}" for k, v in m['segmentation']['dice'].items()
                             if k != "background"))
        st.caption(f"checkpoint `{m['checkpoint']}` · {m['split']} split · {m['n_images']} images · "
                   "⚠️ measured against Stage-1 pseudo-labels, so this is *agreement with "
                   "approximate labels*, not certified accuracy.")
    else:
        st.info("Run `py -3.14 -m src.models.evaluate --split test "
                "--save data/processed/model_metrics.json` to populate this.")
    st.divider()

    # ---- Explanation faithfulness (validation sample) ----
    st.markdown("### 🧪 Explanation faithfulness (validation sample)")
    st.caption("How well the Grad-CAM explanations reflect the model, averaged over a validation "
               "sample — separate from the single-image trust shown in the Explainability tab.")
    xsum = "data/processed/xai/_summary.csv"
    if os.path.exists(xsum):
        xdf = pd.read_csv(xsum)
        ph = xdf["pointing_hit"].dropna()
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Mean trust score", f"{xdf['trust_score'].mean():.2f}",
                  help="overall explanation faithfulness (higher = better)")
        f2.metric("Deletion AUC", f"{xdf['deletion_auc'].mean():.2f}",
                  help="lower = better — hiding the highlighted region should drop the prediction")
        f3.metric("Insertion AUC", f"{xdf['insertion_auc'].mean():.2f}",
                  help="higher = better — restoring that region should recover the prediction")
        f4.metric("Pointing-game", f"{ph.mean()*100:.0f}%" if len(ph) else "—",
                  help="how often the heatmap's peak lands on the true defect")
        split = xdf["split"].iloc[0] if "split" in xdf.columns and len(xdf) else "val"
        st.caption(f"over {len(xdf)} {split} images · ⚠️ pointing-game scored against Stage-1 "
                   "pseudo-labels, so treat as indicative until hand-labels exist.")
    else:
        st.info("Run `py -3.14 -m src.xai.run --config configs/xai.yaml --split val --limit 16` "
                "to populate this.")
    st.divider()

    st.markdown("**How the type is decided**")
    st.write(f"- **Primary (shown everywhere): `{res['defect_type']}`** via the "
             f"**{res.get('type_method','—')}** "
             f"(confidence {res['type_confidence']:.2f})." if res.get("type_confidence") is not None
             else f"- **Primary type: `{res['defect_type']}`** (segmentation-derived).")
    st.write("  The **scattering classifier** uses whole-image wavelet-texture features "
             "(log-scattering + a tiny MLP, the IWSCN recipe) — it reaches "
             "**~0.88 balanced** accuracy (porosity ~0.86, slag ~0.90), fixing the porosity↔slag "
             "confusion the per-pixel approach had (porosity was ~0.50).")
    st.write(f"- *Diagnostic only:* the neural **classification head** says "
             f"`{res['classifier_type']}` (prob {res['classifier_prob']:.2f}) — weaker, not used.")
    st.markdown("**Pipeline**")
    st.code("image → preprocess (S1) → SCN-Attention U-Net (S2) → characterization (S2) "
            "→ Grad-CAM + trust (S3) → digital-twin health (S4)")
    st.markdown("**Measurement scale**")
    st.write(f"pixel→mm = **{res['pixel_to_mm']}** → values shown in **{unit}**. "
             "Scale-free fields (orientation, aspect ratio, eccentricity) are correct regardless.")
    st.json(h)
