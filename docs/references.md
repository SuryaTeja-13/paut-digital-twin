# References

Every work the project actually draws on, grouped by role. The two papers marked **[given]**
were provided directly; the rest are the methods cited across the design/spec/hyperparameter
docs and implemented in the code. BibTeX keys (for the report) are in `docs/references.bib`.

## Core template & survey (provided)

1. **[given]** B. He et al., *"Real-time detection of insulator defects based on improved wavelet
   scattering convolutional network,"* 2025. — The template the architecture transfers from
   insulators to PAUT welds (improved wavelet-scattering + small CNN for small-sample defect
   detection). PDF: `docs/2025_Real-time detection of insulator defects based on.pdf`.
2. **[given]** *"AI-enabled defect detection in industrial products: A comprehensive survey,"*
   2026 review. — Positions the work in the wider automated-defect-detection literature.

## Wavelet scattering (the SCN prior)

3. S. Mallat, *"Group Invariant Scattering,"* Communications on Pure and Applied Mathematics,
   65(10):1331–1398, 2012.
4. J. Bruna and S. Mallat, *"Invariant Scattering Convolution Networks,"* IEEE TPAMI,
   35(8):1872–1886, 2013.
5. J. Andén and S. Mallat, *"Deep Scattering Spectrum,"* IEEE Trans. Signal Processing,
   62(16):4114–4128, 2014.
6. M. Andreux et al., *"Kymatio: Scattering Transforms in Python,"* JMLR, 21(60):1–6, 2020. —
   The scattering implementation used (`src/models/scattering.py`).

## Segmentation backbone & attention

7. O. Ronneberger, P. Fischer, and T. Brox, *"U-Net: Convolutional Networks for Biomedical Image
   Segmentation,"* MICCAI, 2015.
8. O. Oktay et al., *"Attention U-Net: Learning Where to Look for the Pancreas,"* MIDL /
   arXiv:1804.03999, 2018. — The additive attention gates on the skip connections.
9. S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, *"CBAM: Convolutional Block Attention Module,"*
   ECCV, 2018. — The channel+spatial attention in the SCN-fusion blocks.

## Loss functions

10. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, *"Focal Loss for Dense Object
    Detection,"* ICCV, 2017. — The Focal term that down-weights easy background pixels.
11. S. S. M. Salehi, D. Erdogmus, and A. Gholipour, *"Tversky Loss Function for Image Segmentation
    Using 3D Fully Convolutional Deep Networks,"* MLMI (MICCAI workshop), 2017. — The
    foreground-focused Focal-Tversky term that fixed the all-background collapse.

## Explainable AI & faithfulness

12. R. R. Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization,"* ICCV, 2017.
13. A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, *"Grad-CAM++: Generalized
    Gradient-based Visual Explanations for Deep Convolutional Networks,"* WACV, 2018. — The chosen
    XAI method (highest faithfulness in our comparison).
14. M. T. Ribeiro, S. Singh, and C. Guestrin, *"'Why Should I Trust You?': Explaining the
    Predictions of Any Classifier" (LIME),* KDD, 2016.
15. S. M. Lundberg and S.-I. Lee, *"A Unified Approach to Interpreting Model Predictions" (SHAP),*
    NeurIPS, 2017.
16. V. Petsiuk, A. Das, and K. Saenko, *"RISE: Randomized Input Sampling for Explanation of
    Black-box Models,"* BMVC, 2018. — Source of the deletion/insertion AUC faithfulness metrics.

## Characterization & tooling

17. S. van der Walt et al., *"scikit-image: image processing in Python,"* PeerJ, 2:e453, 2014. —
    Connected-components / region properties used for blob analysis (size, orientation via PCA).

## Digital twin

18. F. Tao, H. Zhang, A. Liu, and A. Y. C. Nee, *"Digital Twin in Industry: State-of-the-Art,"*
    IEEE Trans. Industrial Informatics, 15(4):2405–2415, 2019.
19. M. Grieves and J. Vickers, *"Digital Twin: Mitigating Unpredictable, Undesirable Emergent
    Behavior in Complex Systems,"* in *Transdisciplinary Perspectives on Complex Systems*,
    Springer, 2017.

---

*Note:* items 3–19 are the standard primary sources for each technique implemented in the codebase.
If the literature review needs PAUT/TFM domain references beyond these, the specific
inspection standards/papers used in the lab can be added here verbatim — none are
fabricated.
