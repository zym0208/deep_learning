"""Week 3 experiment: does the analysis-driven change help? (small CPU run, not the final training)

    python tools/week3_experiment.py cache/ [--steps 600] [--seeds 0 1 2]

Trains the refiner on the train split of a cache folder (<clip>_s2c.npz + <clip>_gt.npz +
splits.json) with two settings and reports validation errors next to raw and smoothed S2C:
  v1  Week 2 refiner: fingers + wrist position, plain losses
  v2  Week 3 change: + global orientation, depth down-weighted, velocity against a
      smoothed label (see docs/week3_analysis.md)
The real training script (train_refiner.py, teammate) should take over the settings that win.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refiner import LossWeights, MotionRefiner, RefinerConfig, refiner_loss  # noqa: E402
from refiner.baselines import smooth_clip  # noqa: E402
from refiner.motion_format import load_clip  # noqa: E402
from refiner.motion_metrics import axis_angle_to_matrix_np as aa2m, finger_mats, geodesic_deg  # noqa: E402

HANDS = ("right", "left")
FOCAL = 37500.0
VARIANTS = {
    "v1": (RefinerConfig(), LossWeights()),
    "v2": (RefinerConfig(refine_global=True),
           LossWeights(rec_glob=1.0, trans_axis=(1.0, 1.0, 0.25), smooth_target=True)),
}


def load_split(root: Path, names):
    raw, gt = [], []
    for n in names:
        raw.append(load_clip(root / f"{n}_s2c.npz"))
        gt.append(load_clip(root / f"{n}_gt.npz"))
    return raw, gt


def stack(clips, keys):
    return {k: torch.as_tensor(np.stack([c[k] for c in clips])) for k in keys}


def evaluate(pred_clips, gt_clips):
    """Mean over valid frames of all clips x hands."""
    acc = {k: [] for k in ("finger_deg", "global_deg", "wrist_xy_mm", "wrist_z_mm", "image_px",
                           "p95_finger_speed", "finger_jitter")}
    for p, g in zip(pred_clips, gt_clips):
        for h in HANDS:
            v = g[f"valid_{h}"] > 0.5
            if not v.any():
                continue
            pp, gp, pt, gt = p[f"{h}_pose"], g[f"{h}_pose"], p[f"{h}_trans"], g[f"{h}_trans"]
            acc["finger_deg"].append(geodesic_deg(finger_mats(pp[v]), finger_mats(gp[v])).mean())
            acc["global_deg"].append(geodesic_deg(aa2m(pp[v, :3]), aa2m(gp[v, :3])).mean())
            acc["wrist_xy_mm"].append(np.linalg.norm(pt[v, :2] - gt[v, :2], axis=1).mean() * 1000)
            acc["wrist_z_mm"].append(np.abs(pt[v, 2] - gt[v, 2]).mean() * 1000)
            ok = v & (gt[:, 2] > 1)
            proj = lambda t: t[:, :2] / t[:, 2:3] * FOCAL
            acc["image_px"].append(np.linalg.norm(proj(pt[ok]) - proj(gt[ok]), axis=1).mean())
            R = finger_mats(pp); sp = geodesic_deg(R[:-1], R[1:]) * 30
            acc["p95_finger_speed"].append(np.percentile(sp, 95))
            acc["finger_jitter"].append(np.abs(np.diff(sp, axis=0)).mean() * 30)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def run_model(model, clips):
    keys = ("right_pose", "left_pose", "right_trans", "left_trans")
    with torch.no_grad():
        out = model.eval()(stack(clips, keys))
    return [{k: out[k][i].numpy() for k in keys} for i in range(len(clips))]


def train(variant, raw, gt, steps, seed, log_every=100):
    torch.manual_seed(seed)
    cfg, w = VARIANTS[variant]
    model = MotionRefiner(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    keys = ("right_pose", "left_pose", "right_trans", "left_trans")
    X, Y = stack(raw, keys), stack(gt, keys + ("valid_right", "valid_left"))
    n, win, bs = len(raw), 120, 8
    g = torch.Generator().manual_seed(seed)
    curve = []
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), generator=g)
        s0 = int(torch.randint(0, X["right_pose"].shape[1] - win + 1, (1,), generator=g))
        xb = {k: v[idx, s0:s0 + win] for k, v in X.items()}
        yb = {k: v[idx, s0:s0 + win] for k, v in Y.items()}
        model.train()
        out, res = model(xb, return_residual=True)
        loss, parts = refiner_loss(out, yb, res, w)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == steps - 1:
            curve.append((step, loss.item()))
    return model, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--save", help="folder for checkpoints and results.json")
    a = ap.parse_args(argv)
    root = Path(a.cache)
    sp = json.load(open(root / "splits.json"))
    tr_raw, tr_gt = load_split(root, [c["clip"] for c in sp["train"]])
    va_raw, va_gt = load_split(root, [c["clip"] for c in sp["val"]])

    rows = {"raw S2C": [evaluate(va_raw, va_gt)], "smoothed": [evaluate([smooth_clip(c) for c in va_raw], va_gt)]}
    curves = {}
    for v in VARIANTS:
        rows[v] = []
        for seed in a.seeds:
            model, curve = train(v, tr_raw, tr_gt, a.steps, seed)
            curves[f"{v}/seed{seed}"] = curve
            rows[v].append(evaluate(run_model(model, va_raw), va_gt))
            if a.save:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                model.save(str(Path(a.save) / f"refiner_{v}_seed{seed}.pt"), variant=v, steps=a.steps)
    keys = list(rows["raw S2C"][0])
    print(f"validation ({len(va_raw)} clips), mean over {len(a.seeds)} seed(s)")
    print(f"{'':12s}" + "".join(f"{k:>18s}" for k in keys))
    table = {}
    for name, rs in rows.items():
        m = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        table[name] = m
        print(f"{name:12s}" + "".join(f"{m[k]:18.2f}" for k in keys))
    print("training loss (step, loss):", {k: [(s, round(l, 4)) for s, l in c[::max(1, len(c) // 4)]] for k, c in curves.items()})
    if a.save:
        json.dump({"table": table, "curves": curves, "steps": a.steps, "seeds": a.seeds},
                  open(Path(a.save) / "results.json", "w"), indent=1)
    return table


if __name__ == "__main__":
    main()
