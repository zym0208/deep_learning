"""Compare motion versions of one clip (Week 3: raw S2C vs smoothed vs refined).

    python tools/analyze_motion.py raw.npz refined.npz [--smooth] [--gt gt.npz] [--plot out.png]

Each .npz is in the shared clip format (refiner/motion_format.py). --smooth adds
the Savitzky-Golay baseline computed from the first file. --gt is a clip with the
PianoMotion10M annotation for the same frames (from_annotation_json -> save_clip).
Prints a table per hand, the frames with the worst finger jitter (to inspect in
Unity, frame = time * 30), and the change of each version relative to the first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refiner.baselines import smooth_clip  # noqa: E402
from refiner.motion_format import load_clip  # noqa: E402
from refiner.motion_metrics import HANDS, clip_metrics, finger_mats, geodesic_deg, worst_frames  # noqa: E402

LOWER_IS_BETTER = {"wrist_acc_ms2", "wrist_jitter_ms3", "finger_jitter_degs2", "hf_energy_ratio",
                   "wrist_err_mm", "finger_err_deg"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--smooth", action="store_true", help="add savgol(5,2) of the first clip as a version")
    ap.add_argument("--gt")
    ap.add_argument("--plot", help="save per-frame finger speed / wrist error plot (needs matplotlib)")
    a = ap.parse_args(argv)

    versions = [(Path(p).stem, load_clip(p)) for p in a.clips]
    if a.smooth:
        versions.insert(1, ("smoothed", smooth_clip(versions[0][1])))
    gt = load_clip(a.gt) if a.gt else None
    if gt is not None:
        T = min(min(len(c["right_pose"]) for _, c in versions), len(gt["right_pose"]))
        cut = lambda c: {k: (v[:T] if np.ndim(v) else v) for k, v in c.items()}
        versions = [(n, cut(c)) for n, c in versions]; gt = cut(gt)

    results = [(n, clip_metrics(c, gt)) for n, c in versions]
    for h in HANDS:
        keys = list(results[0][1][h])
        print(f"\n== {h} hand ==")
        print(f"{'metric':24s}" + "".join(f"{n[:14]:>16s}" for n, _ in results))
        for k in keys:
            base = results[0][1][h][k]
            row = f"{k:24s}"
            for i, (_, r) in enumerate(results):
                v = r[h][k]
                rel = "" if i == 0 or base == 0 else f" ({(v - base) / abs(base) * 100:+.0f}%)"
                row += f"{v:>9.3g}{rel:>7s}"
            print(row)
    print("\nlower is better for:", ", ".join(sorted(LOWER_IS_BETTER)),
          "\nfinger_speed_p95_degs should stay close to raw (or GT): a big drop means fast attacks were flattened.")

    for n, c in versions:
        print(f"\nworst finger jitter frames in {n} (frame, seconds, deg/s^2):")
        for h, fr in worst_frames(c).items():
            print(f"  {h:5s} " + "  ".join(f"{f}@{t}s" for f, t, _ in fr))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2 if gt is None else 3, 2, figsize=(12, 7 if gt is None else 10), sharex=True)
        for j, h in enumerate(HANDS):
            for n, c in versions:
                fps = float(c.get("fps", 30))
                R = finger_mats(c[f"{h}_pose"]); sp = geodesic_deg(R[:-1], R[1:]).mean(1) * fps
                t = np.arange(1, len(R)) / fps
                ax[0, j].plot(t, sp, lw=.8, label=n)
                w = np.linalg.norm(np.diff(c[f"{h}_trans"], axis=0), axis=-1) * fps
                ax[1, j].plot(t, w, lw=.8, label=n)
                if gt is not None:
                    e = np.linalg.norm(c[f"{h}_trans"] - gt[f"{h}_trans"], axis=-1) * 1000
                    ax[2, j].plot(np.arange(len(e)) / fps, e, lw=.8, label=n)
            ax[0, j].set_title(f"{h}: mean finger speed (deg/s)")
            ax[1, j].set_title(f"{h}: wrist speed (m/s)")
            if gt is not None: ax[2, j].set_title(f"{h}: wrist error vs GT (mm)")
            ax[-1, j].set_xlabel("seconds")
        ax[0, 0].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print(f"\nplot -> {a.plot}")
    return results


if __name__ == "__main__":
    main()
