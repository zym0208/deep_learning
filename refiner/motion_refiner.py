"""MotionRefiner: a small temporal residual network placed after frozen S2C.

    raw S2C motion (both hands, T frames)
        -> normalise -> Conv1d x2 (temporal) -> zero-initialised 1x1 head
        -> bounded residual (tanh * bound)
        -> finger joints: R = R_raw @ exp(delta)   (rotation composition, not addition)
           wrist position: t = t_raw + delta_t * TRANS_SCALE
        -> refined motion, same format as the input

By default the global hand orientation (pose[:, 0:3]) is kept unchanged (version 1).
refine_global=True (Week 3) also corrects it, R_glob = R_raw @ exp(delta): on Parnia's
cache the global orientation is S2C's largest rotation error (17 deg vs 10.6 deg for fingers).
Because the head starts at zero, an untrained refiner returns raw S2C exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import math

import torch
from torch import nn

from .motion_format import FINGER_SLICE, POSE_DIM, TRANS_DIM, TRANS_SCALE
from .rotation import compose_axis_angle

HANDS = ("right", "left")
N_FINGER = FINGER_SLICE.stop - FINGER_SLICE.start  # 45
IN_PER_HAND = POSE_DIM + TRANS_DIM                 # 51
OUT_PER_HAND = N_FINGER + TRANS_DIM                # 48 (+3 with refine_global)


@dataclass
class RefinerConfig:
    hidden: int = 256
    kernel: int = 5
    dilation2: int = 2           # second conv is dilated: receptive field = 1 + 4 + 8 = 13 frames (~0.43 s)
    dropout: float = 0.1
    max_rot: float = 0.25        # rad, max correction per finger-joint axis-angle component
    max_trans: float = 0.05      # in normalised units (x TRANS_SCALE -> 7.5 cm in x/y)
    refine_global: bool = False  # also correct the global hand orientation (pose[:, :3])


class MotionRefiner(nn.Module):
    def __init__(self, cfg: RefinerConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or RefinerConfig()
        self.n_rot = N_FINGER + (3 if cfg.refine_global else 0)
        self.out_per_hand = self.n_rot + TRANS_DIM
        c_in, c_out = 2 * IN_PER_HAND, 2 * self.out_per_hand
        k = cfg.kernel
        self.body = nn.Sequential(
            nn.Conv1d(c_in, cfg.hidden, k, padding=k // 2, padding_mode="replicate"),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(cfg.hidden, cfg.hidden, k, dilation=cfg.dilation2,
                      padding=cfg.dilation2 * (k // 2), padding_mode="replicate"),
            nn.GELU(),
        )
        self.head = nn.Conv1d(cfg.hidden, c_out, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("trans_scale", torch.as_tensor(TRANS_SCALE))

    # ------------------------------------------------------------ helpers
    def _features(self, m: dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, T, 102) -> (B, 102, T), roughly unit-scaled."""
        parts = []
        for h in HANDS:
            parts.append(m[f"{h}_pose"] / math.pi)
            parts.append(m[f"{h}_trans"] / self.trans_scale)
        return torch.cat(parts, -1).transpose(1, 2)

    def predict_residual(self, m: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Bounded corrections per hand: '<h>_drot' (B,T,15,3) rad (16 joints, global first, with
        refine_global), '<h>_dtrans' (B,T,3) normalised."""
        out = self.head(self.body(self._features(m))).transpose(1, 2)  # (B, T, 2 * out_per_hand)
        res, n = {}, self.n_rot
        for i, h in enumerate(HANDS):
            o = out[..., i * self.out_per_hand:(i + 1) * self.out_per_hand]
            res[f"{h}_drot"] = (torch.tanh(o[..., :n]) * self.cfg.max_rot).reshape(*o.shape[:2], n // 3, 3)
            res[f"{h}_dtrans"] = torch.tanh(o[..., n:]) * self.cfg.max_trans
        return res

    # ------------------------------------------------------------ forward
    def forward(self, m: dict[str, torch.Tensor], return_residual: bool = False):
        """m: dict with right/left _pose (B,T,48) and _trans (B,T,3). Returns the same keys refined."""
        res = self.predict_residual(m)
        out = {}
        for h in HANDS:
            pose = m[f"{h}_pose"]
            start = 0 if self.cfg.refine_global else FINGER_SLICE.start
            joints = pose[..., start:].reshape(*pose.shape[:2], -1, 3)
            joints = compose_axis_angle(joints, res[f"{h}_drot"]).reshape(*pose.shape[:2], -1)
            out[f"{h}_pose"] = torch.cat([pose[..., :start], joints], -1)
            out[f"{h}_trans"] = m[f"{h}_trans"] + res[f"{h}_dtrans"] * self.trans_scale
        return (out, res) if return_residual else out

    # ------------------------------------------------------------ io
    def save(self, path: str, **extra) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> "MotionRefiner":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(RefinerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])
        return model
