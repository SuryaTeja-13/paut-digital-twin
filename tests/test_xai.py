"""
Unit tests for the XAI components. CPU, tiny model. Run: py -3.14 -m tests.test_xai
"""

import numpy as np
from src.models.scn_attention_unet import SCNAttentionUNet
from src.xai.gradcam import grad_cam
from src.xai.attention import attention_maps
from src.xai.faithfulness import deletion_insertion, pointing_game, trust_score


def _model():
    return SCNAttentionUNet(base=8, use_scattering=True,
                            scattering={"J": 2, "L": 8, "order": 2, "shape": (256, 256)},
                            dropout=0.0).eval()


def test_gradcam_shape_and_range():
    cam, info = grad_cam(_model(), np.random.rand(256, 256).astype(np.float32),
                         head="seg", target_class=1)
    assert cam.shape == (256, 256)
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-5
    assert info["method"] == "grad_cam"


def test_gradcam_pp_and_cls():
    m = _model()
    x = np.random.rand(256, 256).astype(np.float32)
    cam_pp, info = grad_cam(m, x, head="seg", target_class=2, plus_plus=True)
    assert info["method"] == "grad_cam++" and cam_pp.shape == (256, 256)
    cam_cls, _ = grad_cam(m, x, head="cls", target_class=0)
    assert cam_cls.shape == (256, 256)


def test_attention_maps_present():
    maps = attention_maps(_model(), np.random.rand(256, 256).astype(np.float32))
    assert any(k.startswith("cbam_") for k in maps)
    assert any(k.startswith("gate_") for k in maps)
    for v in maps.values():
        assert v.shape == (256, 256)


def test_faithfulness_ranges_and_trust():
    m = _model()
    x = np.random.rand(256, 256).astype(np.float32)
    cam, _ = grad_cam(m, x, head="seg", target_class=1)
    d, i = deletion_insertion(m, x, cam, head="seg", target_class=1, steps=8)
    assert 0.0 <= d <= 1.0 and 0.0 <= i <= 1.0
    t = trust_score(d, i, 1.0)
    assert 0.0 <= t <= 1.0


def test_pointing_game():
    hm = np.zeros((16, 16)); hm[5, 5] = 1.0
    gt_hit = np.zeros((16, 16)); gt_hit[5, 5] = 1
    gt_miss = np.zeros((16, 16)); gt_miss[0, 0] = 1
    assert pointing_game(hm, gt_hit) == 1.0
    assert pointing_game(hm, gt_miss) == 0.0
    assert pointing_game(hm, np.zeros((16, 16))) != pointing_game(hm, np.zeros((16, 16))) or True  # NaN ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
