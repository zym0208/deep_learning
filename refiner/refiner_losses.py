"""Training losses for MotionRefiner.

    total = w_rec * reconstruction + w_vel * velocity + w_res * residual

reconstruction  refined pose/position closer to the PianoMotion10M label
velocity        frame-to-frame change closer to the label's, so fast key
                strikes are kept (plain smoothing would erase them)
residual        keep corrections small so the network stays close to S2C

Rotations are compared as rotation matrices (no axis-angle wrap-around
problem). Frames whose label is missing (valid == 0) are ignored.
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
    parts = {k: 0.0 for k in ("rec_rot", "rec_trans", "vel_rot", "vel_trans", "res_rot", "res_trans")}
    for h in HANDS:
        mask = target.get(f"valid_{h}")
        vmask = None if mask is None else mask[:, 1:] * mask[:, :-1]

        rp, rt = _finger_mats(pred[f"{h}_pose"]), _finger_mats(target[f"{h}_pose"])
        tp, tt = pred[f"{h}_trans"] / scale, target[f"{h}_trans"] / scale

        parts["rec_rot"] = parts["rec_rot"] + _masked_mean((rp - rt) ** 2, mask)
        parts["rec_trans"] = parts["rec_trans"] + _masked_mean((tp - tt).abs(), mask)

        dvp, dvt = rp[:, 1:] - rp[:, :-1], rt[:, 1:] - rt[:, :-1]
        parts["vel_rot"] = parts["vel_rot"] + _masked_mean((dvp - dvt) ** 2, vmask)
        parts["vel_trans"] = parts["vel_trans"] + _masked_mean(
            ((tp[:, 1:] - tp[:, :-1]) - (tt[:, 1:] - tt[:, :-1])).abs(), vmask)

        parts["res_rot"] = parts["res_rot"] + (residual[f"{h}_drot"] ** 2).mean()
        parts["res_trans"] = parts["res_trans"] + (residual[f"{h}_dtrans"] ** 2).mean()

    total = sum(getattr(w, k) * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}
