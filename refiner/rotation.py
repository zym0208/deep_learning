"""Axis-angle <-> rotation-matrix helpers (pure PyTorch, differentiable).

MANO / S2C store every joint rotation as a 3D axis-angle vector. Adding two
axis-angle vectors is NOT the same as composing the two rotations, so the
refiner turns its predicted correction into a rotation matrix and multiplies:

    R_refined = R_raw @ exp(delta)

and converts back with ``matrix_to_axis_angle`` when it needs MANO input.
"""
from __future__ import annotations

import torch

_EPS = 1e-8


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrix."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([o, -z, y], -1),
            torch.stack([z, o, -x], -1),
            torch.stack([-y, x, o], -1),
        ],
        -2,
    )


def axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula. (..., 3) -> (..., 3, 3).

    Uses Taylor expansions near zero so gradients stay finite at aa == 0
    (which is exactly where a zero-initialised refiner starts).
    """
    theta2 = (aa * aa).sum(-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    theta = torch.sqrt(theta2 + _EPS * _EPS)
    small = theta2 < 1e-6
    # sin(t)/t and (1-cos t)/t^2 with series fallback
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta * theta))
    k = _skew(aa)
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(k.shape)
    return eye + a * k + b * (k @ k)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) quaternion (w, x, y, z), w >= 0. Numerically
    stable branch selection (same idea as PyTorch3D)."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                -1,
            ),
            min=0.0,
        )
        + 1e-12
    )
    cands = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
        ],
        -2,
    )
    cands = cands / (2.0 * q_abs[..., None].clamp(min=0.1))
    idx = q_abs.argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(*idx.shape, 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def quaternion_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    w, xyz = q[..., :1], q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    half = torch.atan2(s, w)
    angle = 2 * half
    small = s < 1e-6
    # angle / sin(angle/2) with series fallback (-> 2 as angle -> 0)
    factor = torch.where(small, 2 + angle * angle / 12, angle / torch.where(small, torch.ones_like(s), torch.sin(half)))
    return xyz * factor


def matrix_to_axis_angle(m: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(m))


def compose_axis_angle(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply ``delta`` in each joint's local frame: R = R_base @ R_delta.
    Both (..., 3); returns (..., 3) axis-angle."""
    return matrix_to_axis_angle(axis_angle_to_matrix(base) @ axis_angle_to_matrix(delta))
