"""The one motion format shared by S2C output, PianoMotion10M labels, the
refiner and the Unity exporter. See docs/refiner_interface.md.

A clip is a dict / .npz with (T = frames at 30 fps):

    right_pose   (T, 48) float32  radians, MANO axis-angle: [0:3] global orient, [3:48] 15 finger joints
    left_pose    (T, 48) float32  same, for the left hand (S2C mirrors a right-hand MANO)
    right_trans  (T, 3)  float32  wrist translation in S2C camera space (metres, OpenCV axes)
    left_trans   (T, 3)  float32
    fps          ()      float    30
    valid_right  (T,)    float32  optional, 1 = hand annotated in this frame (PianoMotion10M column 0)
    valid_left   (T,)    float32  optional

S2C's own infer.py multiplies the network output by pi (poses) and by
TRANS_SCALE (translation). Store the values *after* that scaling and *before*
the Savitzky-Golay smoothing infer.py applies, so "raw S2C" stays raw.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

POSE_DIM = 48          # per hand
FINGER_SLICE = slice(3, 48)   # 15 joints x 3, the part the refiner may change
TRANS_DIM = 3
FPS = 30
# infer.py: scale = torch.tensor([1.5, 1.5, 25]); guide is tanh output in [-1, 1]
TRANS_SCALE = np.array([1.5, 1.5, 25.0], dtype=np.float32)

KEYS = ("right_pose", "left_pose", "right_trans", "left_trans")


def load_clip(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        clip = {k: z[k].astype(np.float32) for k in z.files}
    validate_clip(clip)
    return clip


def save_clip(path: str | Path, clip: dict) -> None:
    validate_clip(clip)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in clip.items()})


def validate_clip(clip: dict) -> None:
    missing = [k for k in KEYS if k not in clip]
    if missing:
        raise KeyError(f"clip is missing {missing}")
    t = clip["right_pose"].shape[0]
    for k, d in zip(KEYS, (POSE_DIM, POSE_DIM, TRANS_DIM, TRANS_DIM)):
        if clip[k].shape != (t, d):
            raise ValueError(f"{k} has shape {clip[k].shape}, expected ({t}, {d})")
        if not np.isfinite(clip[k]).all():
            raise ValueError(f"{k} contains NaN/inf")


def from_annotation_json(path: str | Path, adjust: bool = True) -> dict[str, np.ndarray]:
    """Read one PianoMotion10M annotation file (e.g. BV1f34y1i7U1_seq_0000.json).

    Each row of mano['right'] / mano['left'] is
    [valid, tx, ty, tz, 48 pose, 10 betas]; PianoPose.py adds mano['adjust'][1:3]
    to (ty, tz) when --adjust is on (S2C was trained that way).
    Note: annotation fps may differ from 30; resample before training if so.
    """
    with open(path) as f:
        mano = json.load(f)
    out = {"fps": np.float32(mano["fps"])}
    for hand in ("right", "left"):
        a = np.asarray(mano[hand], dtype=np.float32)
        if adjust:
            a[:, 2:4] += np.asarray(mano["adjust"][1:3], dtype=np.float32)
        out[f"valid_{hand}"] = a[:, 0]
        out[f"{hand}_trans"] = a[:, 1:4]
        out[f"{hand}_pose"] = a[:, 4:52]
    return out


# ---------------------------------------------------------------- tensors

def clip_to_tensors(clip: dict, device="cpu") -> dict[str, torch.Tensor]:
    """numpy clip -> batched torch tensors of shape (1, T, D)."""
    return {
        k: torch.as_tensor(np.asarray(clip[k]), dtype=torch.float32, device=device).unsqueeze(0)
        for k in clip
        if k != "fps" and np.asarray(clip[k]).ndim >= 1
    }
