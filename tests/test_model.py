"""
Shape / wiring tests for the SCN-Attention U-Net. CPU, tiny tensors -> fast.
Run:  py -3.14 -m tests.test_model
"""

import torch

from src.models.scn_attention_unet import SCNAttentionUNet
from src.models.losses import MultiTaskLoss
from src.models.metrics import SegAccumulator


def _model(use_scattering=True, base=8):
    # base=8 keeps it tiny/fast; scattering shape must match input (256)
    return SCNAttentionUNet(base=base, use_scattering=use_scattering,
                            scattering={"J": 2, "L": 8, "order": 2, "shape": (256, 256)},
                            dropout=0.0)


def test_forward_shapes_with_scattering():
    m = _model(True).eval()
    x = torch.randn(2, 1, 256, 256)
    out = m(x, return_features=True)
    assert out["seg"].shape == (2, 3, 256, 256)
    assert out["cls"].shape == (2, 2)
    assert out["decoder_last"].shape[-2:] == (256, 256)


def test_forward_shapes_baseline():
    m = _model(False).eval()
    out = m(torch.randn(1, 1, 256, 256))
    assert out["seg"].shape == (1, 3, 256, 256)
    assert out["cls"].shape == (1, 2)


def test_scattering_is_frozen():
    m = _model(True)
    n_scat_trainable = sum(p.requires_grad for p in m.scat.parameters())
    assert n_scat_trainable == 0          # scattering filters must not train


def test_loss_runs_and_is_finite():
    m = _model(True).train()
    x = torch.randn(2, 1, 256, 256)
    mask = torch.randint(0, 3, (2, 256, 256))
    label = torch.randint(0, 2, (2,))
    out = m(x)
    crit = MultiTaskLoss(seg_classes=3)
    loss, parts = crit(out, mask, label)
    assert torch.isfinite(loss)
    loss.backward()                        # gradients flow
    assert all(k in parts for k in ("dice", "focal", "boundary", "cls"))


def test_backward_does_not_touch_scattering():
    m = _model(True).train()
    out = m(torch.randn(1, 1, 256, 256))
    mask = torch.randint(0, 3, (1, 256, 256))
    label = torch.randint(0, 2, (1,))
    loss, _ = MultiTaskLoss(seg_classes=3)(out, mask, label)
    loss.backward()
    grads = [p.grad for p in m.scat.parameters() if p.grad is not None]
    assert len(grads) == 0                 # no grad reached the fixed filters


def test_loss_punishes_all_background_collapse():
    # Guard against the imbalance collapse: predicting all-background must cost
    # MORE than predicting the (tiny) foreground correctly.
    crit = MultiTaskLoss(seg_classes=3)
    mask = torch.zeros(1, 16, 16, dtype=torch.long)
    mask[0, 7:9, 7:9] = 1                       # a tiny porosity blob
    label = torch.zeros(1, dtype=torch.long)

    big = 12.0
    bg_logits = torch.zeros(1, 3, 16, 16); bg_logits[:, 0] = big      # all background
    ok_logits = torch.zeros(1, 3, 16, 16); ok_logits[:, 0] = big
    ok_logits[0, 0, 7:9, 7:9] = -big; ok_logits[0, 1, 7:9, 7:9] = big  # correct blob
    cls = torch.tensor([[5.0, -5.0]])

    l_bg, _ = crit({"seg": bg_logits, "cls": cls}, mask, label)
    l_ok, _ = crit({"seg": ok_logits, "cls": cls}, mask, label)
    assert l_ok < l_bg, "loss should reward segmenting the defect over all-background"


def test_seg_accumulator():
    acc = SegAccumulator(3)
    pred = torch.tensor([[[0, 1], [2, 1]]])
    acc.update(pred, pred)                 # perfect prediction
    dice, iou = acc.finalize()
    assert all(d > 0.99 for d in dice) and all(i > 0.99 for i in iou)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
