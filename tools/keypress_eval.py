"""MIDI key presses for Unity, and the key-press check for the comparison table.

1) Notes file for Unity (MidiKeyboard.cs), no MANO needed:
    python tools/keypress_eval.py notes <data_root>/midi/<up>/<video>/<seq>.mid \
           --duration 8 -o unity/Assets/StreamingAssets/<seq>_notes.json

2) Key-press metric on cached clips (needs smplx + MANO, so run on the HPC):
    python tools/keypress_eval.py eval cache/ --data-root <pianomotion> --mano-root <dir with mano/MANO_RIGHT.pkl>
   For each clip in cache/splits.json: read its MIDI (first 8 s, as cached), fit the keyboard
   on the GT fingertips, score GT and every <clip>_<variant>.npz with that keyboard.
   Prints hit rate (fingertip within half a key at the onset) and mean distance in mm.
   See refiner/keypress.py for the method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refiner.keypress import (WHITE_KEY_MM, fingertips_from_clip, fit_keyboard,  # noqa: E402
                              image_plane, keypress_scores, read_midi_notes)
from refiner.motion_format import load_clip  # noqa: E402


def cmd_notes(a):
    notes = read_midi_notes(a.midi, a.start, a.duration)
    Path(a.out).write_text(json.dumps({"source": Path(a.midi).name, "start": a.start, "notes": notes}))
    print(f"{len(notes)} notes -> {a.out}")


def cmd_eval(a):
    cache = Path(a.cache)
    splits = json.loads((cache / "splits.json").read_text())
    rows = []
    for split in a.splits:
        for item in splits.get(split, []):
            clip, up, vid = item["clip"], item["up"], item["video"]
            midi = Path(a.data_root) / "midi" / up / vid / f"{clip}.mid"
            if not midi.exists():
                print("no midi:", midi); continue
            notes = read_midi_notes(midi, 0.0, a.duration)
            gt = load_clip(cache / f"{clip}_gt.npz")
            gt_tips = image_plane(fingertips_from_clip(gt, a.mano_root))
            kb = fit_keyboard(gt_tips, notes, 30)
            if kb is None:
                print("too few notes:", clip); continue
            row = {"clip": clip, "split": split, "onsets": kb["n_onsets"], "fit_rms_keys": kb["rms_keys"]}
            for var in ["gt"] + a.variants:
                f = cache / f"{clip}_{var}.npz"
                if not f.exists():
                    continue
                tips = gt_tips if var == "gt" else image_plane(fingertips_from_clip(load_clip(f), a.mano_root))
                s = keypress_scores(tips, notes, 30, kb, a.tol)
                row[f"{var}_hit"] = float(s["hit"].mean())
                row[f"{var}_mm"] = float(s["dist_keys"].mean() * WHITE_KEY_MM)
            rows.append(row)
            print(json.dumps(row))
    if not rows:
        return
    print(f"\nmean over {len(rows)} clips (hit = fingertip within {a.tol} key of the key at onset)")
    for var in ["gt"] + a.variants:
        h = [r[f"{var}_hit"] for r in rows if f"{var}_hit" in r]
        mm = [r[f"{var}_mm"] for r in rows if f"{var}_mm" in r]
        if h:
            print(f"  {var:10s} hit rate {np.mean(h):.3f}   distance {np.mean(mm):.1f} mm   ({len(h)} clips)")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("notes")
    n.add_argument("midi")
    n.add_argument("--start", type=float, default=0.0)
    n.add_argument("--duration", type=float, default=None)
    n.add_argument("-o", "--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("cache")
    e.add_argument("--data-root", required=True)
    e.add_argument("--mano-root", required=True)
    e.add_argument("--variants", nargs="+", default=["s2c"], help="e.g. s2c smooth refined ft")
    e.add_argument("--splits", nargs="+", default=["val", "test"])
    e.add_argument("--duration", type=float, default=8.0)
    e.add_argument("--tol", type=float, default=0.5)
    e.add_argument("--json")
    a = ap.parse_args()
    cmd_notes(a) if a.cmd == "notes" else cmd_eval(a)


if __name__ == "__main__":
    main()
