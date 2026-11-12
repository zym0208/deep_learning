"""Baseline: the same Savitzky-Golay smoothing S2C's infer.py applies
(window 5, order 2, per channel). The refiner must beat this, otherwise it is
just an expensive filter."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .motion_format import KEYS


def smooth_clip(clip: dict, window: int = 5, order: int = 2) -> dict:
    out = dict(clip)
    for k in KEYS:
        out[k] = savgol_filter(np.asarray(clip[k]), window, order, axis=0).astype(np.float32)
    return out
