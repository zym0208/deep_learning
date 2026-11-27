"""Key presses from MIDI: fit where the keyboard is, then check whether a fingertip is on
the right key at every note onset. Pure numpy (MIDI reading needs `mido`, which S2C
already uses). The same fit runs inside Unity (unity/Assets/Scripts/MidiKeyboard.cs).

Why a fit: PianoMotion10M gives hands in camera space and MIDI per clip, but no keyboard
position. Every note onset says "some fingertip is over key k now", so the keyboard is the
line  p(k) = o + u(k) * v  in the image plane that best explains the onsets, where u(k) is
the key's position counted in white keys (black keys sit halfway between two whites) and
v is one white-key step. We do not know which of the 10 fingertips played each note, so
the fit alternates (1-D ICP):

    1. guess o, v from the spread of the fingertips (both directions tried),
    2. assign each onset to its nearest fingertip,
    3. least-squares refit of o, v; drop onsets more than 2 keys away; repeat.

We use image-plane coordinates (x/z, y/z) because S2C's depth z is unreliable (see
docs/week3_analysis.md) and the camera is far away, so the keyboard is a straight line
there. Distances come out in key widths and are turned into mm with the real white-key
width (23.5 mm), so the camera scale does not matter.

Metrics (evaluate_keypress):
    key_hit_rate       share of onsets with a fingertip within `tol` key widths of the key
                       centre, along the keyboard. tol=0.5 means "on the right key".
    key_dist_keys      mean along-keyboard distance of the nearest fingertip, key widths
    key_dist_mm        the same in mm (x 23.5)
Fit the keyboard on the ground truth of a clip, then score GT and prediction with that same
keyboard. GT's own hit rate tells how good the labels are (upper bound for any model).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# MANO 778-vertex fingertips (smplx vertex_ids['mano']); the left hand uses the same ids.
MANO_TIP_IDS = (744, 320, 443, 554, 671)        # thumb, index, middle, ring, pinky
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
WHITE_KEY_MM = 23.5
_BLACK = {1, 3, 6, 8, 10}


def is_black(pitch: int) -> bool:
    return pitch % 12 in _BLACK


def key_coord(pitch) -> np.ndarray:
    """Key position in white-key units (C4 = 60 -> 0, D4 -> 1, C#4 -> 0.5)."""
    p = np.atleast_1d(np.asarray(pitch, dtype=int))
    octave, pc = np.divmod(p - 60, 12)
    white_in_octave = np.array([0, 0.5, 1, 1.5, 2, 3, 3.5, 4, 4.5, 5, 5.5, 6])
    return octave * 7 + white_in_octave[pc]


# ------------------------------------------------------------------ MIDI

def read_midi_notes(path: str | Path, start: float = 0.0, duration: float | None = None):
    """Notes of a MIDI file as a list of {pitch, on, off, velocity}, seconds relative to
    `start`. Notes overlapping [start, start+duration] are kept and clipped to it.
    PianoMotion10M: <data_root>/midi/<up>/<video>/<seq>.mid, cached clips use start=0, 8 s."""
    import mido

    t, open_, notes = 0.0, {}, []
    for msg in mido.MidiFile(str(path)):          # iterating a MidiFile gives seconds, tempo-aware
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_.setdefault(msg.note, []).append((t, msg.velocity))
        elif msg.type in ("note_off", "note_on") and open_.get(msg.note):
            on, vel = open_[msg.note].pop(0)
            notes.append((msg.note, on, t, vel))
    for pitch, lst in open_.items():               # unterminated notes end with the file
        notes += [(pitch, on, t, vel) for on, vel in lst]
    end = np.inf if duration is None else start + duration
    out = [{"pitch": int(p), "on": round(max(on, start) - start, 4),
            "off": round(min(off, end) - start, 4), "velocity": int(v)}
           for p, on, off, v in sorted(notes, key=lambda n: n[1]) if off > start and on < end]
    return out


def onsets(notes, fps: float, n_frames: int):
    """(pitch array, frame array) for notes whose onset lies inside the clip."""
    p = np.array([n["pitch"] for n in notes if 0 <= n["on"] * fps < n_frames], dtype=int)
    f = np.array([int(round(n["on"] * fps)) for n in notes if 0 <= n["on"] * fps < n_frames],
                 dtype=int)
    return p, np.minimum(f, n_frames - 1)


# ------------------------------------------------------------------ fingertips

def fingertips_from_clip(clip: dict, mano_root: str | Path) -> np.ndarray:
    """(T, 10, 3) camera-space fingertips, right hand first (thumb..pinky), via MANO
    (same decoding as tools/export_unity.py / S2C show.py). NaN where a hand is invalid."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.export_unity import mano_meshes

    vr, _, vl = mano_meshes(clip, mano_root)
    tips = np.concatenate([vr[:, MANO_TIP_IDS], vl[:, MANO_TIP_IDS]], 1).astype(np.float64)
    for i, hand in enumerate(("right", "left")):
        if f"valid_{hand}" in clip:
            tips[np.asarray(clip[f"valid_{hand}"]) < 0.5, 5 * i:5 * i + 5] = np.nan
    return tips


def image_plane(tips: np.ndarray) -> np.ndarray:
    """(…, 3) camera points -> (…, 2) normalised image coordinates x/z, y/z."""
    return tips[..., :2] / tips[..., 2:3]


# ------------------------------------------------------------------ keyboard fit

def _fit_line(u, pts, w):
    """Weighted least squares pts ≈ o + u v. Returns o (2,), v (2,)."""
    A = np.stack([np.ones_like(u), u], 1) * w[:, None]
    sol, *_ = np.linalg.lstsq(A, pts * w[:, None], rcond=None)
    return sol[0], sol[1]


def _nearest(cands, centre, v):
    """Per onset: index of the fingertip nearest along the keyboard, and that distance in keys."""
    step = np.linalg.norm(v)
    along = np.abs(((cands - centre[:, None]) @ (v / step))) / step   # (N, 10)
    along = np.where(np.isnan(along), np.inf, along)
    idx = along.argmin(1)
    return idx, along[np.arange(len(idx)), idx]


def _robust_cost(cands, u, o, v, cap=0.5):
    _, d = _nearest(cands, o + u[:, None] * v, v)
    return np.mean(np.minimum(d, cap) ** 2), d


def _polish(cands, u, o, v):
    best = (_robust_cost(cands, u, o, v)[0], o, v)
    um = np.median(u)
    for scale in np.linspace(0.9, 1.1, 9):
        for shift in np.linspace(-1.0, 1.0, 41):
            v2 = v * scale
            o2 = o + um * (v - v2) + shift * v          # scale about the median key
            c = _robust_cost(cands, u, o2, v2)[0]
            if c < best[0]:
                best = (c, o2, v2)
    _, o, v = best
    for _ in range(3):
        idx, d = _nearest(cands, o + u[:, None] * v, v)
        keep = d < 0.5
        if keep.sum() < 4:
            break
        o, v = _fit_line(u[keep], cands[np.arange(len(u)), idx][keep], np.ones(keep.sum()))
    return o, v


def fit_keyboard(tips2d: np.ndarray, notes, fps: float, iters: int = 15):
    """tips2d (T, 10, 2) fingertips in the plane where the keyboard is a line (image plane).
    Returns dict(o, v, rms_keys, n_used, n_onsets) or None if there are < 4 usable onsets."""
    pitch, frame = onsets(notes, fps, len(tips2d))
    cands = tips2d[frame]
    ok = ~np.all(np.isnan(cands[..., 0]), 1)
    pitch, cands = pitch[ok], cands[ok]
    if len(pitch) < 4 or np.ptp(key_coord(pitch)) < 1:
        return None
    u = key_coord(pitch)
    flat = cands.reshape(-1, 2)
    flat = flat[~np.isnan(flat[:, 0])]
    mean = flat.mean(0)
    axis = np.linalg.svd(flat - mean, full_matrices=False)[2][0]        # main spread direction
    s = (flat - mean) @ axis
    # hands span more than the played keys; match the 10-90 % ranges for a first scale
    scale = (np.percentile(s, 90) - np.percentile(s, 10)) / max(np.percentile(u, 90) - np.percentile(u, 10), 1.0)
    best = None
    for sign in (1.0, -1.0):
        v = axis * scale * sign
        o = mean - np.median(u) * v
        keep = np.ones(len(u), bool)
        for it in range(iters):
            idx, _ = _nearest(cands, o + u[:, None] * v, v)
            pts = cands[np.arange(len(u)), idx]
            if keep.sum() < 4:
                break
            o, v = _fit_line(u[keep], pts[keep], np.ones(keep.sum()))
            _, d = _nearest(cands, o + u[:, None] * v, v)
            keep = d < (2.0 if it >= 2 else np.inf)
        # fingers are about one key apart, so ICP can lock on one finger off: polish with a
        # small grid over shift / scale under a tight robust cost, then refit on the inliers
        o, v = _polish(cands, u, o, v)
        _, d = _nearest(cands, o + u[:, None] * v, v)
        cost = np.mean(np.minimum(d, 2.0) ** 2)
        if best is None or cost < best[0]:
            best = (cost, o, v, int((d < 2.0).sum()))
    cost, o, v, n_used = best
    return {"o": o, "v": v, "rms_keys": float(np.sqrt(cost)), "n_used": n_used, "n_onsets": len(u)}


def keypress_scores(tips2d: np.ndarray, notes, fps: float, kb: dict, tol: float = 0.5):
    """Per onset distances to the key with a fitted keyboard `kb` (from fit_keyboard)."""
    pitch, frame = onsets(notes, fps, len(tips2d))
    cands = tips2d[frame]
    ok = ~np.all(np.isnan(cands[..., 0]), 1)
    pitch, frame, cands = pitch[ok], frame[ok], cands[ok]
    centre = kb["o"] + key_coord(pitch)[:, None] * kb["v"]
    idx, d = _nearest(cands, centre, kb["v"])
    return {"pitch": pitch, "frame": frame, "tip": idx, "dist_keys": d, "hit": d < tol}


def evaluate_keypress(gt_tips2d, notes, fps, pred_tips2d=None, tol: float = 0.5):
    """Fit on GT, score GT (and prediction) with the same keyboard. Returns a flat dict."""
    kb = fit_keyboard(gt_tips2d, notes, fps)
    if kb is None:
        return {}
    out = {"kb_rms_keys": kb["rms_keys"], "n_onsets": kb["n_onsets"]}
    for name, tips in (("gt", gt_tips2d), ("pred", pred_tips2d)):
        if tips is None:
            continue
        s = keypress_scores(tips, notes, fps, kb, tol)
        out[f"{name}_key_hit_rate"] = float(s["hit"].mean())
        out[f"{name}_key_dist_keys"] = float(s["dist_keys"].mean())
        out[f"{name}_key_dist_mm"] = float(s["dist_keys"].mean() * WHITE_KEY_MM)
    return out
