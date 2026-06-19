"""Student 3: explainable AI (Seg-Grad-CAM, intrinsic attention maps, faithfulness).

SHAP was evaluated for the classification head (see docs/architecture.md §6) but is
NOT implemented — the class head is routed around (seg-derived type), so the
implemented XAI is gradient CAMs on the seg head + intrinsic attention + faithfulness.
"""
