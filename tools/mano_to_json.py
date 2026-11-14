"""Convert MANO_RIGHT.pkl into mano_right.json for demo/piano-hands.html.

The page then builds real MANO hand meshes from S2C output (.npz) in the
browser, with the same math as smplx (zero betas, flat_hand_mean=False,
pose correctives, linear blend skinning) and S2C's left-hand mirroring.

    pip install smplx torch numpy
    python tools/mano_to_json.py --mano-root <folder containing mano/MANO_RIGHT.pkl> -o mano_right.json

MANO is licensed for non-commercial research only. The JSON is derived from
it: keep it on your own machines, do not commit it or publish it. The page
reads it from a local file you pick; it never uploads it anywhere.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mano-root", required=True, help="folder that contains mano/MANO_RIGHT.pkl (as smplx.create expects)")
    ap.add_argument("-o", "--out", default="mano_right.json")
    args = ap.parse_args()

    import smplx
    import torch

    m = smplx.create(args.mano_root, "mano", use_pca=False, is_rhand=True)  # same call as S2C
    v = m.v_template.detach().cpu().numpy().astype(np.float64)               # (778, 3)
    J = (m.J_regressor.detach().cpu().numpy() @ v)                            # (16, 3), betas = 0
    posedirs = m.posedirs.detach().cpu().numpy()                              # (135, 2334)
    weights = m.lbs_weights.detach().cpu().numpy()                            # (778, 16)
    parents = m.parents.detach().cpu().numpy().astype(int).tolist()
    parents[0] = -1
    pose_mean = m.pose_mean.detach().cpu().numpy()                            # (48,) = 0,0,0 + hands_mean
    faces = np.asarray(m.faces, dtype=np.int64)

    r = lambda a, d=6: np.round(np.asarray(a, dtype=np.float64), d).ravel().tolist()
    out = {
        "format": "mano-right-lbs-1",
        "note": "Derived from MANO (research-only licence). Do not publish.",
        "v_template": r(v), "J": r(J), "posedirs": r(posedirs, 6), "weights": r(weights, 5),
        "parents": parents, "pose_mean": r(pose_mean), "faces": faces.ravel().tolist(),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    # self-check against smplx on a random pose so the browser math has a reference
    torch.manual_seed(0)
    pose = torch.randn(1, 48) * 0.3
    with torch.no_grad():
        ref = m(global_orient=pose[:, :3], hand_pose=pose[:, 3:], betas=torch.zeros(1, 10)).vertices[0].numpy()
    print(f"wrote {args.out}; check vertex[0] for pose seed 0 = {ref[0].round(5).tolist()}")


if __name__ == "__main__":
    main()
