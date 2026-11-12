"""Turn a motion clip into a .hmotion file the Unity HandMotionPlayer can play.

Real MANO meshes (needs smplx + MANO_RIGHT.pkl, same as S2C):
    python tools/export_unity.py --clip cache/s2c_raw/clip_0000.npz --mano-root /path/to/S2C -o clip_0000.hmotion
    python tools/export_unity.py --annotation PianoMotion10M/annotation/<up>/<bv>/<bv>_seq_0000.json --mano-root ... -o gt.hmotion

No MANO yet (tests the Unity player with two block hands):
    python tools/export_unity.py --dummy -o dummy.hmotion

--mano-root is the folder that contains mano/MANO_RIGHT.pkl (S2C calls smplx.create('./', 'mano')).

Mesh decoding follows S2C datasets/show.py exactly: right MANO model for both
hands, zero betas, left hand = right-hand mesh with x negated.

Coordinates: S2C works in OpenCV camera space (x right, y down, z forward,
right-handed). Unity is y-up, left-handed, so we flip y. Flipping one axis
mirrors the mesh, so the right hand's triangle winding is reversed; the left
hand is mirrored twice (x then y) and keeps MANO's winding.
By default the clip is re-centred on the mean wrist position, so a Unity
camera at (0, 0, -1) looking down +z sees both hands.

File layout (little-endian):
    char[4] 'HMOT'; int32 version=1; float32 fps; int32 frames;
    int32 nVertR, nTriR, nVertL, nTriL;
    int32 trisR[nTriR*3]; int32 trisL[nTriL*3];
    frames x (float32 vertsR[nVertR*3], float32 vertsL[nVertL*3])
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refiner.motion_format import FPS, from_annotation_json, load_clip  # noqa: E402

MAGIC, VERSION = b"HMOT", 1


def write_hmotion(path, fps, verts_r, tris_r, verts_l, tris_l):
    verts_r = np.ascontiguousarray(verts_r, dtype="<f4")
    verts_l = np.ascontiguousarray(verts_l, dtype="<f4")
    assert verts_r.shape[0] == verts_l.shape[0]
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ifiiiii", VERSION, float(fps), verts_r.shape[0],
                            verts_r.shape[1], len(tris_r), verts_l.shape[1], len(tris_l)))
        f.write(np.asarray(tris_r, dtype="<i4").tobytes())
        f.write(np.asarray(tris_l, dtype="<i4").tobytes())
        for a, b in zip(verts_r, verts_l):
            f.write(a.tobytes())
            f.write(b.tobytes())


def to_unity(verts_r, faces, verts_l, recenter=None):
    """OpenCV camera space -> Unity. verts: (T, V, 3)."""
    flip = np.array([1, -1, 1], dtype=np.float32)
    vr, vl = verts_r * flip, verts_l * flip
    if recenter is not None:
        c = recenter * flip
        vr, vl = vr - c, vl - c
    tris_r = faces[:, ::-1].copy()  # mirrored once  -> reverse winding
    tris_l = faces.copy()           # mirrored twice -> keep winding
    return vr, tris_r, vl, tris_l


def mano_meshes(clip, mano_root, device="cpu"):
    import smplx
    import torch

    layer = smplx.create(str(mano_root), "mano", use_pca=False, is_rhand=True).to(device)
    out = {}
    for hand in ("right", "left"):
        pose = torch.as_tensor(clip[f"{hand}_pose"], dtype=torch.float32, device=device)
        trans = torch.as_tensor(clip[f"{hand}_trans"], dtype=torch.float32, device=device)
        with torch.no_grad():
            v = layer(global_orient=pose[:, :3], hand_pose=pose[:, 3:48],
                      betas=torch.zeros(len(pose), 10, device=device), transl=trans).vertices
        v = v.cpu().numpy()
        if hand == "left":
            v[:, :, 0] *= -1
        out[hand] = v
    return out["right"], np.asarray(layer.faces, dtype=np.int64), out["left"]


# ------------------------------------------------------------------ dummy hands

def _box(center, size):
    c, s = np.asarray(center, np.float32), np.asarray(size, np.float32) / 2
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * s + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def dummy_meshes(seconds=8.0, fps=FPS):
    """Two block hands (palm + 5 fingers) in OpenCV camera space, each finger
    pressing in turn, so the Unity player can be tested without MANO/S2C."""
    parts, faces, n = [], [], 0
    for c, s in [((0, 0, 0), (0.08, 0.02, 0.09))] + [
            ((x, 0, 0.075), (0.014, 0.016, 0.06)) for x in (-0.035, -0.017, 0.0, 0.017, 0.035)]:
        v, f = _box(c, s)
        parts.append(v); faces.append(f + n); n += len(v)
    faces = np.concatenate(faces)
    T = int(seconds * fps)
    out = []
    for hand_x in (0.15, -0.15):  # right hand at +x (dummy is already 'mirrored' for left by placement)
        frames = []
        for t in range(T):
            time = t / fps
            verts = []
            for i, v in enumerate(parts):
                v = v.copy()
                if i > 0:  # finger i presses when its turn comes (y down = towards keys in camera space)
                    press = max(0.0, np.sin(2 * np.pi * (time * 1.5 - (i - 1) / 5))) ** 4
                    ang = 0.6 * press
                    rel = v - np.array([0, 0, 0.045], np.float32)
                    rot = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]], np.float32)
                    v = rel @ rot.T + np.array([0, 0, 0.045], np.float32)
                verts.append(v)
            v = np.concatenate(verts)
            v += np.array([hand_x + 0.03 * np.sin(time), 0, 1.0], np.float32)
            frames.append(v)
        out.append(np.stack(frames))
    return out[0], faces, out[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="motion .npz in the refiner format")
    src.add_argument("--annotation", help="PianoMotion10M annotation .json")
    src.add_argument("--dummy", action="store_true", help="block hands, no MANO needed")
    ap.add_argument("--mano-root", help="folder containing mano/MANO_RIGHT.pkl")
    ap.add_argument("--no-recenter", action="store_true")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    if args.dummy:
        vr, faces, vl = dummy_meshes()
        fps = FPS
        # neither dummy hand is mirrored, so after the y flip both need reversed winding
        vr, _, vl, _ = to_unity(vr, faces, vl, recenter=np.array([0, 0, 1.0], np.float32))
        tris_r = tris_l = faces[:, ::-1].copy()
    else:
        if not args.mano_root:
            ap.error("--mano-root is required unless --dummy")
        clip = load_clip(args.clip) if args.clip else from_annotation_json(args.annotation)
        fps = float(clip.get("fps", FPS))
        vr, faces, vl = mano_meshes(clip, args.mano_root)
        recenter = None if args.no_recenter else np.concatenate(
            [clip["right_trans"], clip["left_trans"]]).mean(0).astype(np.float32)
        vr, tris_r, vl, tris_l = to_unity(vr, faces, vl, recenter)

    write_hmotion(args.out, fps, vr, tris_r.reshape(-1, 3), vl, tris_l.reshape(-1, 3))
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {vr.shape[0]} frames @ {fps} fps, {vr.shape[1]}+{vl.shape[1]} verts, {size:.1f} MB")


if __name__ == "__main__":
    main()
