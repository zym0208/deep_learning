"""Week 2 acceptance checks for the refiner (run on CPU: `pytest -q`).

Uses synthetic motion with the exact S2C shapes, so it runs before any
S2C output or PianoMotion10M data exists. The teammate's real cached clips
can be dropped in later by replacing `make_batch`.
"""
import math

import pytest
import torch

from refiner import LossWeights, MotionRefiner, RefinerConfig, refiner_loss
from refiner.motion_format import FINGER_SLICE, TRANS_SCALE
from refiner.rotation import (axis_angle_to_matrix, compose_axis_angle,
                              matrix_to_axis_angle)

torch.manual_seed(0)
B, T = 4, 240  # 8 s at 30 fps, like S2C's train_sec=8


def make_batch(b=B, t=T, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    time = torch.linspace(0, 8, t)
    m = {}
    for h in ("right", "left"):
        phase = torch.rand(b, 1, 48, generator=g) * 6.28
        pose = 0.4 * torch.sin(2 * time[None, :, None] + phase)
        trans = torch.as_tensor(TRANS_SCALE) * 0.3 * torch.sin(time[None, :, None] * 0.5 + phase[..., :3])
        m[f"{h}_pose"] = pose + noise * torch.randn(pose.shape, generator=g)
        m[f"{h}_trans"] = trans + noise * 0.05 * torch.randn(trans.shape, generator=g)
    return m


# ---------------------------------------------------------------- rotations

def test_rotation_round_trip():
    axis = torch.nn.functional.normalize(torch.randn(1000, 3), dim=-1)
    aa = axis * torch.rand(1000, 1) * 3.0           # angles in [0, 3) rad, below pi
    back = matrix_to_axis_angle(axis_angle_to_matrix(aa))
    assert torch.allclose(back, aa, atol=1e-4)


def test_compose_is_not_addition():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    composed = axis_angle_to_matrix(compose_axis_angle(a, b))
    expected = axis_angle_to_matrix(a) @ axis_angle_to_matrix(b)
    assert torch.allclose(composed, expected, atol=1e-5)
    assert not torch.allclose(composed, axis_angle_to_matrix(a + b), atol=1e-2)


def test_rotation_grad_finite_at_zero():
    aa = torch.zeros(5, 3, requires_grad=True)
    axis_angle_to_matrix(aa).sum().backward()
    assert torch.isfinite(aa.grad).all()


# ---------------------------------------------------------------- refiner

def test_zero_init_returns_raw_s2c():
    model = MotionRefiner().eval()
    m = make_batch()
    with torch.no_grad():
        out, res = model(m, return_residual=True)
    for k in res:
        assert res[k].abs().max() == 0
    for h in ("right", "left"):
        assert torch.allclose(out[f"{h}_pose"], m[f"{h}_pose"], atol=1e-4)
        assert torch.equal(out[f"{h}_trans"], m[f"{h}_trans"])


def test_global_orientation_never_changes():
    model = MotionRefiner()
    with torch.no_grad():
        model.head.bias.fill_(0.5)   # force large corrections
    m = make_batch()
    out = model(m)
    for h in ("right", "left"):
        assert torch.equal(out[f"{h}_pose"][..., :3], m[f"{h}_pose"][..., :3])
        assert not torch.allclose(out[f"{h}_pose"][..., FINGER_SLICE], m[f"{h}_pose"][..., FINGER_SLICE])


def test_residual_is_bounded():
    cfg = RefinerConfig(max_rot=0.1, max_trans=0.02)
    model = MotionRefiner(cfg)
    with torch.no_grad():
        model.head.bias.fill_(100.0)
    _, res = model(make_batch(), return_residual=True)
    for h in ("right", "left"):
        assert res[f"{h}_drot"].abs().max() <= 0.1 + 1e-6
        assert res[f"{h}_dtrans"].abs().max() <= 0.02 + 1e-6


def test_gradients_flow_and_weights_change():
    model = MotionRefiner(RefinerConfig(dropout=0.0))
    raw, gt = make_batch(noise=0.05, seed=1), make_batch(seed=1)
    out, res = model(raw, return_residual=True)
    loss, _ = refiner_loss(out, gt, res)
    loss.backward()
    # zero-init head: the head gets gradient on step 0, the body only after the head moves
    assert model.head.weight.grad.abs().sum() > 0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out, res = model(raw, return_residual=True)
        refiner_loss(out, gt, res)[0].backward()
        opt.step()
    for n, p in model.named_parameters():
        assert not torch.equal(before[n], p), f"{n} did not change"


def test_frozen_upstream_gets_no_gradient():
    """S2C stays frozen: inputs produced under no_grad carry no grad, and the
    refiner never asks for one."""
    model = MotionRefiner()
    raw = {k: v.requires_grad_(False) for k, v in make_batch().items()}
    out, res = model(raw, return_residual=True)
    refiner_loss(out, make_batch(), res)[0].backward()
    assert all(v.grad is None for v in raw.values())


def test_save_load_identical(tmp_path):
    model = MotionRefiner(RefinerConfig(hidden=64, dropout=0.0))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    path = tmp_path / "refiner.pt"
    model.save(str(path), step=7)
    loaded = MotionRefiner.load(str(path))
    assert loaded.cfg == model.cfg
    m = make_batch(b=1)
    model.eval(); loaded.eval()
    with torch.no_grad():
        a, b = model(m), loaded(m)
    for k in a:
        assert torch.equal(a[k], b[k])


def test_masked_frames_are_ignored():
    model = MotionRefiner()
    raw = make_batch(noise=0.1, seed=2)
    gt = make_batch(seed=2)
    gt_bad = {k: v.clone() for k, v in gt.items()}
    for h in ("right", "left"):
        gt_bad[f"{h}_pose"][:, 100:] = 9.0   # garbage label ...
        mask = torch.ones(B, T); mask[:, 100:] = 0  # ... but marked invalid
        gt[f"valid_{h}"] = mask; gt_bad[f"valid_{h}"] = mask
    out, res = model(raw, return_residual=True)
    l1, _ = refiner_loss(out, gt, res)
    l2, _ = refiner_loss(out, gt_bad, res)
    assert torch.allclose(l1, l2)


def test_overfits_one_batch():
    """The implementation can learn: loss on one fixed batch must drop a lot."""
    torch.manual_seed(0)
    model = MotionRefiner(RefinerConfig(hidden=128, dropout=0.0))
    raw, gt = make_batch(b=2, noise=0.08, seed=3), make_batch(b=2, seed=3)
    w = LossWeights(res_rot=0.0, res_trans=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    with torch.no_grad():
        out, res = model(raw, return_residual=True)
        first = refiner_loss(out, gt, res, w)[0].item()
    for _ in range(300):
        opt.zero_grad()
        out, res = model(raw, return_residual=True)
        loss, _ = refiner_loss(out, gt, res, w)
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first, (first, loss.item())


def test_week3_refine_global_and_loss_options():
    model = MotionRefiner(RefinerConfig(refine_global=True)).eval()
    m = make_batch()
    with torch.no_grad():
        out = model(m)
    for h in ("right", "left"):
        assert torch.allclose(out[f"{h}_pose"], m[f"{h}_pose"], atol=1e-4)   # zero init still = raw
    with torch.no_grad():
        model.head.bias.fill_(0.5)
        out, res = model(m, return_residual=True)
    assert res["right_drot"].shape[-2] == 16
    assert not torch.allclose(out["right_pose"][..., :3], m["right_pose"][..., :3])
    w = LossWeights(rec_glob=1.0, trans_axis=(1.0, 1.0, 0.25), smooth_target=True)
    loss, parts = refiner_loss(out, make_batch(seed=3), res, w)
    assert torch.isfinite(loss) and parts["rec_glob"] > 0
