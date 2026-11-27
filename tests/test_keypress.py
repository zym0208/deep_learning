"""Key-press metric: the keyboard fit recovers a known keyboard from synthetic onsets."""
import numpy as np
import pytest

from refiner.keypress import evaluate_keypress, fit_keyboard, key_coord, keypress_scores


def synth(seed=0, n_notes=60, T=240, fps=30, angle=0.2, flip=False, noise=0.05):
    """Ten fingertips hovering over a tilted keyboard; one tip lands on each note's key."""
    r = np.random.default_rng(seed)
    step = 0.03 * np.array([np.cos(angle), np.sin(angle)]) * (-1 if flip else 1)
    o = np.array([0.1, -0.05])
    perp = np.array([-step[1], step[0]])
    t = np.arange(T)
    # hands slowly move; right hand around keys +3..+7, left around -9..-5 (white-key units)
    centre_r = 5 + 2 * np.sin(t / 40)
    centre_l = -7 + 2 * np.cos(t / 50)
    offs = np.array([-2, -1, 0, 1, 2], float)
    keys = np.concatenate([centre_r[:, None] + offs, centre_l[:, None] + offs], 1)   # (T, 10)
    keys += 0.15 * r.standard_normal(keys.shape)
    tips = o + keys[..., None] * step + 0.5 * r.standard_normal((T, 10, 1)) * perp
    notes, frames = [], r.choice(np.arange(5, T - 5), n_notes, replace=False)
    for f in sorted(frames):
        j = r.integers(10)
        # snap that tip onto a real key near it
        cand = np.arange(36, 97)
        pitch = int(cand[np.abs(key_coord(cand) - keys[f, j]).argmin()])
        tips[f, j] = o + (key_coord(pitch)[0] + noise * r.standard_normal()) * step
        notes.append({"pitch": pitch, "on": f / fps, "off": f / fps + 0.2, "velocity": 64})
    return tips, notes, o, step


@pytest.mark.parametrize("flip", [False, True])
def test_fit_recovers_keyboard(flip):
    tips, notes, o, step = synth(flip=flip)
    kb = fit_keyboard(tips, notes, 30)
    assert kb is not None
    assert np.allclose(kb["v"], step, atol=0.1 * np.linalg.norm(step))
    s = keypress_scores(tips, notes, 30, kb)
    assert s["hit"].mean() > 0.95


def test_wrong_hands_score_lower():
    tips, notes, *_ = synth(seed=1)
    shifted = tips.copy()
    shifted[..., 0] += 0.045            # 1.5 keys sideways
    m = evaluate_keypress(tips, notes, 30, pred_tips2d=shifted)
    assert m["gt_key_hit_rate"] > 0.95
    assert m["pred_key_hit_rate"] < 0.6
    assert m["pred_key_dist_mm"] > m["gt_key_dist_mm"]


def test_key_coord():
    assert key_coord(60)[0] == 0 and key_coord(62)[0] == 1 and key_coord(61)[0] == 0.5
    assert key_coord(72)[0] == 7 and key_coord(59)[0] == -1


def test_read_midi(tmp_path):
    mido = pytest.importorskip("mido")
    from refiner.keypress import read_midi_notes
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))       # 120 bpm: 480 ticks = 0.5 s
    tr.append(mido.Message("note_on", note=60, velocity=80, time=0))
    tr.append(mido.Message("note_on", note=60, velocity=0, time=480))   # off at 0.5 s
    tr.append(mido.Message("note_on", note=64, velocity=70, time=480))  # on at 1.0 s
    tr.append(mido.Message("note_off", note=64, velocity=0, time=960))  # off at 2.0 s
    mid.save(tmp_path / "a.mid")
    n = read_midi_notes(tmp_path / "a.mid")
    assert [(x["pitch"], x["on"], x["off"]) for x in n] == [(60, 0.0, 0.5), (64, 1.0, 2.0)]
    n = read_midi_notes(tmp_path / "a.mid", start=0.75, duration=0.75)
    assert [(x["pitch"], x["on"], x["off"]) for x in n] == [(64, 0.25, 0.75)]
