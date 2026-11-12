"""MotionRefiner: a small temporal residual network placed after frozen S2C.

    raw S2C motion (both hands, T frames)
        -> normalise -> Conv1d x2 (temporal) -> zero-initialised 1x1 head
        -> bounded residual (tanh * bound)
        -> finger joints: R = R_raw @ exp(delta)   (rotation composition, not addition)
           wrist position: t = t_raw + delta_t * TRANS_SCALE
        -> refined motion, same format as the input

Version 1 keeps the global hand orientation (pose[:, 0:3]) unchanged.
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
OUT_PER_HAND = N_FINGER + TRANS_DIM                # 48


@dataclass
class RefinerConfig:
    hidden: int = 256
    kernel: int = 5
    dilation2: int = 2           # second conv is dilated: receptive field = 1 + 4 + 8 = 13 frames (~0.43 s)
    dropout: float = 0.1
    max_rot: float = 0.25        # rad, max correction per finger-joint axis-angle component
    max_trans: float = 0.05      # in normalised units (x TRANS_SCALE -> 7.5 cm in x/y)


class MotionRefiner(nn.Module):
    def __init__(self, cfg: RefinerConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or RefinerConfig()
        c_in, c_out = 2 * IN_PER_HAND, 2 * OUT_PER_HAND
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
        """Bounded corrections per hand: '<h>_drot' (B,T,15,3) rad, '<h>_dtrans' (B,T,3) normalised."""
        out = self.head(self.body(self._features(m))).transpose(1, 2)  # (B, T, 96)
        res = {}
        for i, h in enumerate(HANDS):
            o = out[..., i * OUT_PER_HAND:(i + 1) * OUT_PER_HAND]
            res[f"{h}_drot"] = (torch.tanh(o[..., :N_FINGER]) * self.cfg.max_rot).reshape(*o.shape[:2], 15, 3)
            res[f"{h}_dtrans"] = torch.tanh(o[..., N_FINGER:]) * self.cfg.max_trans
        return res

    # ------------------------------------------------------------ forward
    def forward(self, m: dict[str, torch.Tensor], return_residual: bool = False):
        """m: dict with right/left _pose (B,T,48) and _trans (B,T,3). Returns the same keys refined."""
        res = self.predict_residual(m)
        out = {}
        for h in HANDS:
            pose = m[f"{h}_pose"]
            fingers = pose[..., FINGER_SLICE].reshape(*pose.shape[:2], 15, 3)
            fingers = compose_axis_angle(fingers, res[f"{h}_drot"]).reshape(*pose.shape[:2], N_FINGER)
            out[f"{h}_pose"] = torch.cat([pose[..., :FINGER_SLICE.start], fingers], -1)
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
