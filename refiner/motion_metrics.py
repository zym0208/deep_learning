"""Week 3 diagnostics: measure wrist drift, finger jitter and whether a correction
removes fast piano motion. Pure numpy, works on clips in the shared format
(refiner/motion_format.py): {hand}_pose (T,48), {hand}_trans (T,3), fps.

Definitions (per hand, units in the result keys):
  wrist_jitter_ms3      mean |third difference| of wrist translation (jerk, m/s^3)
  wrist_acc_ms2         mean |second difference| of wrist translation (m/s^2)
  finger_jitter_degs2   mean angular acceleration of the 15 finger joints (deg/s^2),
                        from geodesic angles between consecutive relative rotations
  finger_speed_p95_degs 95th percentile finger angular speed (deg/s): the fast keystrokes.
                        A refiner that cuts this a lot is flattening real attacks.
  hf_energy_ratio       share of finger-angle signal energy above 8 Hz (mostly jitter;
                        piano finger motion is mostly below that at 30 fps)
With a ground-truth clip:
  wrist_err_mm          mean wrist position error
  wrist_drift_mm_per_s  slope of wrist error over time (positive = error grows = drift)
  finger_err_deg        mean geodesic finger-joint rotation error
"""
from __future__ import annotations

import numpy as np

from .motion_format import FINGER_SLICE

HANDS = ("right", "left")


def axis_angle_to_matrix_np(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-12)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*aa.shape[:-1], 3, 3)
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Angle (degrees) of Ra^T Rb, batched over leading dims."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)   # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def finger_mats(pose: np.ndarray) -> np.ndarray:
    """(T,48) -> (T,15,3,3) finger joint rotations."""
    return axis_angle_to_matrix_np(pose[:, FINGER_SLICE].reshape(len(pose), 15, 3))


def hand_metrics(pose, trans, fps=30.0, gt_pose=None, gt_trans=None, valid=None) -> dict:
    pose, trans = np.asarray(pose, np.float64), np.asarray(trans, np.float64)
    T = len(pose)
    out = {}
    if T >= 4:
        d2 = np.diff(trans, 2, axis=0) * fps ** 2
        d3 = np.diff(trans, 3, axis=0) * fps ** 3
        out["wrist_acc_ms2"] = float(np.linalg.norm(d2, axis=-1).mean())
        out["wrist_jitter_ms3"] = float(np.linalg.norm(d3, axis=-1).mean())
        R = finger_mats(pose)
        speed = geodesic_deg(R[:-1], R[1:]) * fps                      # (T-1, 15) deg/s
        out["finger_speed_p95_degs"] = float(np.percentile(speed, 95))
        out["finger_jitter_degs2"] = float(np.abs(np.diff(speed, axis=0)).mean() * fps)
        ang = pose[:, FINGER_SLICE] - pose[:, FINGER_SLICE].mean(0)
        spec = np.abs(np.fft.rfft(ang, axis=0)) ** 2
        f = np.fft.rfftfreq(T, 1 / fps)
        out["hf_energy_ratio"] = float(spec[f > 8].sum() / max(spec[f > 0].sum(), 1e-12))
    if gt_pose is not None:
        m = np.ones(T, bool) if valid is None else np.asarray(valid) > 0.5
        if not m.any():   # hand never annotated in this clip
            return {**out, "wrist_err_mm": np.nan, "wrist_drift_mm_per_s": np.nan, "finger_err_deg": np.nan}
        err = np.linalg.norm(trans - gt_trans, axis=-1) * 1000
        out["wrist_err_mm"] = float(err[m].mean())
        t = np.arange(T)[m] / fps
        out["wrist_drift_mm_per_s"] = float(np.polyfit(t, err[m], 1)[0]) if m.sum() > 2 else 0.0
        out["finger_err_deg"] = float(geodesic_deg(finger_mats(pose)[m], finger_mats(np.asarray(gt_pose))[m]).mean())
    return out


def clip_metrics(clip: dict, gt: dict | None = None) -> dict:
    fps = float(clip.get("fps", 30.0))
    res = {}
    for h in HANDS:
        res[h] = hand_metrics(
            clip[f"{h}_pose"], clip[f"{h}_trans"], fps,
            None if gt is None else gt[f"{h}_pose"], None if gt is None else gt[f"{h}_trans"],
            None if gt is None else gt.get(f"valid_{h}"))
    return res


def worst_frames(clip: dict, k: int = 5) -> dict:
    """Frames with the largest finger angular acceleration per hand: where to look in Unity."""
    fps = float(clip.get("fps", 30.0))
    out = {}
    for h in HANDS:
        R = finger_mats(np.asarray(clip[f"{h}_pose"], np.float64))
        speed = geodesic_deg(R[:-1], R[1:]).max(1) * fps
        acc = np.abs(np.diff(speed)) * fps
        idx = np.argsort(acc)[::-1][:k] + 1
        out[h] = [(int(i), round(i / fps, 2), float(acc[i - 1])) for i in idx]
    return out
