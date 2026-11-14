import argparse, json, os, time
import numpy as np, torch
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
from datasets.PianoPose import PianoPose
from models.piano2posi import Piano2Posi
from models.denoise_diffusion import GaussianDiffusion1D_piano2pose, Unet1D

p = argparse.ArgumentParser()
p.add_argument('--exp_path', default='checkpoints/s2c/piano2pose')
p.add_argument('--data_root', required=True)
p.add_argument('--mode', default='test')
p.add_argument('--n_clips', type=int, default=28)
p.add_argument('--out', default='cache')
a = p.parse_args()
torch.manual_seed(0)

args = argparse.Namespace(**json.load(open(a.exp_path + '/args.txt')))
args.preload, args.data_root, args.up_list, args.is_random = False, a.data_root, [], False
ds = PianoPose(args=args, phase=a.mode)
ap = argparse.Namespace(**json.load(open(a.exp_path + '/args_posi.txt')))
ap.hidden_type = getattr(args, 'hidden_type', 'audio_f')
cond_dim = 768 if 'base' in ap.wav2vec_path else 1024

def build(posi):
    unet = Unet1D(dim=args.unet_dim, dim_mults=(1, 2, 4, 8), channels=args.bs_dim // 2,
                  remap_noise=getattr(args, 'remap_noise', True), condition=True,
                  guide=args.xyz_guide, guide_dim=6 if args.xyz_guide else 0,
                  condition_dim=cond_dim, encoder_type=getattr(args, 'encoder_type', 'none'),
                  num_layer=getattr(args, 'num_layer', None), feature_fusion=args.fusion)
    return GaussianDiffusion1D_piano2pose(unet, posi, seq_length=args.train_sec * 30,
                                          timesteps=args.timesteps, objective='pred_v')

model_left, model_right = build(Piano2Posi(ap)), build(Piano2Posi(ap))
for m, side in [(model_left, 'left'), (model_right, 'right')]:
    m.load_state_dict(torch.load(f'{a.exp_path}/piano2pose-{side}.ckpt', map_location='cpu')['state_dict'])
    m.to('cuda').eval()

scale = torch.tensor([1.5, 1.5, 25]).cuda()
half = args.bs_dim // 2
os.makedirs(a.out, exist_ok=True)
done, seen_videos = [], set()
with torch.no_grad():
    for i, (up, vid, seq) in enumerate(ds.file_list):
        if len(done) == a.n_clips:
            break
        if vid in seen_videos:          # one clip per video -> split by video
            continue
        try:
            t0 = time.time()
            batch, _ = ds.__getitem__(i, True)
            if len(batch['audio']) < args.train_sec * 16000 or batch['right'].shape[0] != args.train_sec * 30:
                print('skip (too short):', seq); continue
            for k in batch:
                batch[k] = torch.as_tensor(batch[k]).cuda().unsqueeze(0)
            audio, rp, lp = batch['audio'], batch['right'], batch['left']
            gt_l = lp[:, :, 4:].permute(0, 2, 1) / np.pi
            gt_r = rp[:, :, 4:].permute(0, 2, 1) / np.pi
            g_l, c_l = model_left.getGuideAndCond(gt_l, audio=audio)
            g_r, c_r = model_right.getGuideAndCond(gt_r, audio=audio)
            p_l, p_r = model_left.sample(torch.cat([g_l, g_r], 2), c_l, torch.cat([g_r, g_l], 2), c_r,
                                         model_right.model, 1)
            pose = torch.cat([p_r, p_l], 1).permute(0, 2, 1)[0]
            guide = torch.cat([g_r, g_l], 2)[0]
            f = lambda x: x.float().cpu().numpy()
            np.savez(f'{a.out}/{seq}_s2c.npz', right_pose=f(pose[:, :half] * np.pi),
                     left_pose=f(pose[:, half:] * np.pi), right_trans=f(guide[:, :3] * scale),
                     left_trans=f(guide[:, 3:] * scale), fps=30)
            np.savez(f'{a.out}/{seq}_gt.npz', right_pose=f(rp[0, :, 4:]), left_pose=f(lp[0, :, 4:]),
                     right_trans=f(rp[0, :, 1:4]), left_trans=f(lp[0, :, 1:4]),
                     valid_right=f(rp[0, :, 0]), valid_left=f(lp[0, :, 0]), fps=30)
            seen_videos.add(vid)
            done.append({'clip': seq, 'up': up, 'video': vid, 'frames': [0, args.train_sec * 30]})
            print(f'[{len(done)}/{a.n_clips}] {seq} done in {time.time() - t0:.1f}s', flush=True)
        except Exception as e:
            print('skip (error):', seq, repr(e)[:200], flush=True)

n_tr = len(done) - 8
splits = {'train': done[:n_tr], 'val': done[n_tr:n_tr + 4], 'test': done[n_tr + 4:],
          'note': f'S2C {a.mode} split, first 8 s of each clip, one clip per video, raw (unsmoothed) S2C output'}
json.dump(splits, open(f'{a.out}/splits.json', 'w'), indent=1)
print('saved', len(done), 'pairs to', a.out, '| train/val/test =', n_tr, 4, 4)
