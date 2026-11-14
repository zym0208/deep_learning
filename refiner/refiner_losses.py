"""Training losses for MotionRefiner.

    total = w_rec * reconstruction + w_vel * velocity + w_res * residual

reconstruction  refined pose/position closer to the PianoMotion10M label
velocity        frame-to-frame change closer to the label's, so fast key
                strikes are kept (plain smoothing would erase them)
residual        keep corrections small so the network stays close to S2C

Rotations are compared as rotation matrices (no axis-angle wrap-around
problem). Frames whose label is missing (valid == 0) are ignored.

Week 3 options, from the analysis of Parnia's cache (docs/week3_analysis.md):
  rec_glob        global hand orientation term (only has gradient with refine_global)
  trans_axis      per-axis weight of the wrist position terms; depth (z) of the
                  far-away S2C camera is poorly observed and its labels are noisy
  smooth_target   compare velocities with a Savitzky-Golay(5,2)-smoothed label, because
                  the labels jitter more than S2C (annotation noise is not motion)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .motion_format import FINGER_SLICE, TRANS_SCALE
from .motion_refiner import HANDS
from .rotation import axis_angle_to_matrix


@dataclass
class LossWeights:
    rec_rot: float = 1.0
    rec_trans: float = 1.0
    vel_rot: float = 1.0
    vel_trans: float = 1.0
    res_rot: float = 0.1
    res_trans: float = 0.1
    rec_glob: float = 0.0
    trans_axis: tuple = (1.0, 1.0, 1.0)
    smooth_target: bool = False


_SG5 = (-3 / 35, 12 / 35, 17 / 35, 12 / 35, -3 / 35)   # savgol_filter(window=5, order=2) centre weights


def _smooth_time(x: torch.Tensor) -> torch.Tensor:
    """Savitzky-Golay(5,2) along dim 1 of (B, T, ...), edges replicated."""
    pad = torch.cat([x[:, :1], x[:, :1], x, x[:, -1:], x[:, -1:]], 1)
    return sum(c * pad[:, i:i + x.shape[1]] for i, c in enumerate(_SG5))


def _glob_mat(pose: torch.Tensor) -> torch.Tensor:
    return axis_angle_to_matrix(pose[..., :3])


def _finger_mats(pose: torch.Tensor) -> torch.Tensor:
    """(B, T, 48) -> (B, T, 15, 3, 3)"""
    return axis_angle_to_matrix(pose[..., FINGER_SLICE].reshape(*pose.shape[:2], 15, 3))


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """x: (B, T, ...) ; mask: (B, T) or None."""
    x = x.reshape(*x.shape[:2], -1).mean(-1)
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def refiner_loss(pred: dict, target: dict, residual: dict, w: LossWeights | None = None):
    """pred/target: dicts with <hand>_pose (B,T,48), <hand>_trans (B,T,3),
    optional target['valid_<hand>'] (B,T). residual: from MotionRefiner(..., return_residual=True).
    Returns (total_loss, dict of detached parts for logging)."""
    w = w or LossWeights()
    scale = torch.as_tensor(TRANS_SCALE, device=pred["right_trans"].device)
    axis = torch.as_tensor(w.trans_axis, device=scale.device, dtype=scale.dtype)
    parts = {k: 0.0 for k in ("rec_rot", "rec_trans", "vel_rot", "vel_trans", "res_rot", "res_trans", "rec_glob")}
    for h in HANDS:
        mask = target.get(f"valid_{h}")
        vmask = None if mask is None else mask[:, 1:] * mask[:, :-1]

        rp, rt = _finger_mats(pred[f"{h}_pose"]), _finger_mats(target[f"{h}_pose"])
        tp, tt = pred[f"{h}_trans"] / scale, target[f"{h}_trans"] / scale

        parts["rec_rot"] = parts["rec_rot"] + _masked_mean((rp - rt) ** 2, mask)
        parts["rec_trans"] = parts["rec_trans"] + _masked_mean((tp - tt).abs() * axis, mask)
        parts["rec_glob"] = parts["rec_glob"] + _masked_mean(
            (_glob_mat(pred[f"{h}_pose"]) - _glob_mat(target[f"{h}_pose"])) ** 2, mask)

        rv, tv = (_smooth_time(rt), _smooth_time(tt)) if w.smooth_target else (rt, tt)
        dvp, dvt = rp[:, 1:] - rp[:, :-1], rv[:, 1:] - rv[:, :-1]
        parts["vel_rot"] = parts["vel_rot"] + _masked_mean((dvp - dvt) ** 2, vmask)
        parts["vel_trans"] = parts["vel_trans"] + _masked_mean(
            ((tp[:, 1:] - tp[:, :-1]) - (tv[:, 1:] - tv[:, :-1])).abs() * axis, vmask)

        parts["res_rot"] = parts["res_rot"] + (residual[f"{h}_drot"] ** 2).mean()
        parts["res_trans"] = parts["res_trans"] + (residual[f"{h}_dtrans"] ** 2).mean()

    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
