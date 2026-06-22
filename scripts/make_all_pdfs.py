"""make_all_pdfs.py — rebuild every project PDF from its Markdown source.

One command regenerates all the document PDFs in docs/ with the plain
black-on-white style (see md_to_pdf.py). The solution-approach PDF embeds the
ablation curve, so it is built by its own script.

Run:  py -3.14 scripts/make_all_pdfs.py
"""
from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.md_to_pdf import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (source markdown, output PDF, title, subtitle)
DOCS = [
    ("docs/architecture.md", "docs/PAUT_Architecture.pdf",
     "System Architecture", "End-to-end design, Student 1 → Student 4"),
    ("design_decisions.md", "docs/PAUT_Design_Decisions.pdf",
     "Design Decisions", "Rationale behind the key engineering choices"),
    ("docs/hyperparameters.md", "docs/PAUT_Hyperparameters.pdf",
     "Hyperparameters", "Settings, values, and their justification"),
    ("docs/model_spec.md", "docs/PAUT_Model_Spec.pdf",
     "Model Specification", "SCN-Attention U-Net — layers, shapes, parameters"),
    ("docs/multidefect_fusion.md", "docs/PAUT_MultiDefect_Fusion.pdf",
     "Multi-Defect Fusion", "Detecting porosity and slag together"),
    ("docs/robustness.md", "docs/PAUT_Robustness.pdf",
     "Robustness Study", "Noise, synthetic defects, and out-of-distribution probes"),
    ("docs/training_improvements.md", "docs/PAUT_Training_Improvements.pdf",
     "Training Improvements", "Augmentation changes and their measured effect"),
    ("docs/xai_method_comparison.md", "docs/PAUT_XAI_Comparison.pdf",
     "Explainable-AI Method Comparison", "Grad-CAM, Grad-CAM++, LIME, SHAP by faithfulness"),
]


def main():
    for md, out, title, sub in DOCS:
        mp = os.path.join(ROOT, md)
        op = os.path.join(ROOT, out)
        if not os.path.exists(mp):
            print("skip (missing):", md)
            continue
        build(mp, op, title, sub, "PAUT Explainable-AI Digital Twin")

    # solution approach has its own builder (embeds the ablation curve)
    print("building solution approach ...")
    runpy.run_path(os.path.join(ROOT, "scripts", "make_solution_pdf.py"), run_name="__main__")
    print("\nall PDFs rebuilt.")


if __name__ == "__main__":
    main()
