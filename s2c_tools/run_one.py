# Run S2C on test clips without rendering/evaluation; save raw (unsmoothed) motion.
import sys, types, numpy as np, torch
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
def save(path, audio, right, left, *rest):
    out = path.rsplit('/pic', 1)[0] + '/motion_raw.npz'
    np.savez(out, right_trans=right[:, :3], right_pose=right[:, 3:51],
             left_trans=left[:, :3], left_pose=left[:, 3:51], fps=30)
    print('saved', out, right.shape)
show = types.ModuleType('datasets.show'); show.render_result = save
sys.modules['datasets.show'] = show
ev = types.ModuleType('models.evaluate')
for n in ['fid', 'loc_distance', 'MW2', 'acceleration', 'acceleration_2']:
    setattr(ev, n, lambda *a, **k: 0.0)
ev.EmbeddingSpaceEvaluator = None
sys.modules['models.evaluate'] = ev
import infer
infer.savgol_filter = lambda x, *a, **k: x
infer.main()
