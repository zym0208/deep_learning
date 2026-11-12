"""The .hmotion writer produces what HandMotionPlayer.cs reads."""
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_hmotion(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"HMOT"
        ver, fps, frames, nvr, ntr, nvl, ntl = struct.unpack("<ifiiiii", f.read(28))
        tr = np.frombuffer(f.read(ntr * 12), "<i4").reshape(-1, 3)
        tl = np.frombuffer(f.read(ntl * 12), "<i4").reshape(-1, 3)
        data = np.frombuffer(f.read(), "<f4").reshape(frames, nvr + nvl, 3)
    return ver, fps, tr, tl, data[:, :nvr], data[:, nvr:]


def test_dummy_export_round_trip(tmp_path):
    out = tmp_path / "dummy.hmotion"
    subprocess.run([sys.executable, str(ROOT / "tools/export_unity.py"), "--dummy", "-o", str(out)], check=True)
    ver, fps, tr, tl, vr, vl = read_hmotion(out)
    assert ver == 1 and fps == 30 and vr.shape == (240, 48, 3) and vl.shape == (240, 48, 3)
    assert tr.max() < 48 and tl.max() < 48
    assert np.isfinite(vr).all() and np.isfinite(vl).all()
    assert vr[:, :, 0].mean() > 0 > vl[:, :, 0].mean()       # right hand on +x in Unity
    assert abs(vr[:, :, 2].mean()) < 0.2                     # re-centred near the origin
