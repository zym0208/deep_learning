"""Week 3 diagnostics: jitter/drift metrics react the right way on synthetic clips."""
import numpy as np

from refiner.baselines import smooth_clip
from refiner.motion_format import save_clip
from refiner.motion_metrics import clip_metrics, worst_frames


def make_clip(T=240, noise=0.0, drift=0.0, seed=0):
    r = np.random.default_rng(seed)
    t = np.arange(T) / 30
    c = {"fps": np.float32(30)}
    for h in ("right", "left"):
        pose = 0.3 * np.sin(2 * np.pi * 1.5 * t[:, None] + r.uniform(0, 6, 48))
        trans = np.stack([0.1 * np.sin(t), 0.05 * np.cos(t), 0.6 + 0 * t], 1) + drift * t[:, None]
        c[f"{h}_pose"] = (pose + noise * r.standard_normal(pose.shape)).astype(np.float32)
        c[f"{h}_trans"] = (trans + noise * 0.01 * r.standard_normal(trans.shape)).astype(np.float32)
    return c


def test_jitter_metrics_rise_with_noise_and_fall_with_smoothing():
    clean, noisy = clip_metrics(make_clip())["right"], clip_metrics(make_clip(noise=0.05))["right"]
    smooth = clip_metrics(smooth_clip(make_clip(noise=0.05)))["right"]
    for k in ("finger_jitter_degs2", "wrist_jitter_ms3", "hf_energy_ratio"):
        assert noisy[k] > clean[k] and smooth[k] < noisy[k], k


def test_drift_and_error_against_ground_truth():
    gt = make_clip()
    m = clip_metrics(make_clip(drift=0.01), gt)["right"]   # 1 cm/s along each axis
    assert abs(m["wrist_drift_mm_per_s"] - 10 * np.sqrt(3)) < 0.5
    assert m["finger_err_deg"] < 1e-3
    assert clip_metrics(gt, gt)["left"]["wrist_err_mm"] < 1e-6


def test_worst_frames_finds_a_spike():
    c = make_clip()
    c["right_pose"][100, 3:48] += 0.5
    assert 100 in [f for f, _, _ in worst_frames(c, k=3)["right"]] or 101 in [f for f, _, _ in worst_frames(c, k=3)["right"]]


def test_cli_runs(tmp_path, capsys):
    from tools.analyze_motion import main
    save_clip(tmp_path / "raw.npz", make_clip(noise=0.05))
    save_clip(tmp_path / "gt.npz", make_clip())
    main([str(tmp_path / "raw.npz"), "--smooth", "--gt", str(tmp_path / "gt.npz")])
    out = capsys.readouterr().out
    assert "finger_jitter_degs2" in out and "smoothed" in out
