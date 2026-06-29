"""Student 3: explainable AI (Seg-Grad-CAM, intrinsic attention maps, faithfulness).

SHAP was evaluated for the classification head (see docs/architecture.md §6) but is
NOT implemented — the class (neural) head is routed around in favour of the
scattering-feature type classifier, so the implemented XAI is gradient CAMs
(Grad-CAM++ / Seg-Grad-CAM) on the segmentation head + intrinsic attention +
faithfulness.
"""
