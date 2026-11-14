"""Week 3: error analysis over a whole cache folder (Parnia's cache/: <clip>_s2c.npz + <clip>_gt.npz + splits.json).

    python tools/analyze_cache.py cache/

Prints, averaged over clips x hands: jitter / speed metrics for raw S2C, savgol-smoothed
S2C and the ground truth, plus errors against the ground truth, split per translation
axis and in image pixels (S2C's renderer: focal 37500 px, see S2C datasets/show.py).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from refiner.baselines import smooth_clip  # noqa: E402
from refiner.motion_format import load_clip  # noqa: E402
from refiner.motion_metrics import axis_angle_to_matrix_np as aa2m, clip_metrics, geodesic_deg  # noqa: E402

FOCAL = 37500.0
root = Path(sys.argv[1] if len(sys.argv) > 1 else "cache")
sp = json.load(open(root / "splits.json"))
rows={}; ax_err={'s2c':[], 'smooth':[]}; proj={'s2c':[],'smooth':[]}; wrot={'s2c':[],'smooth':[]}; gz=[]; empty=[]
P = lambda t: t[:,:2]/t[:,2:3]*FOCAL
for split in ['train','val','test']:
  for c in sp[split]:
    n=c['clip']; s=load_clip(f'{root}/{n}_s2c.npz'); g=load_clip(f'{root}/{n}_gt.npz'); sm=smooth_clip(s)
    for name,cl in [('s2c',s),('smooth',sm),('gt',g)]:
      m=clip_metrics(cl, g if name!='gt' else None)
      for h in m:
        for k,v in m[h].items(): rows.setdefault((name,k),[]).append(v)
    for h in ['right','left']:
      v=g[f'valid_{h}']>0.5
      if not v.any(): empty.append((n,h)); continue
      gz.append((g[f'{h}_trans'][v,2].min(),g[f'{h}_trans'][v,2].max(),g[f'{h}_trans'][v,2].std(), s[f'{h}_trans'][:,2].std()))
      for name,cl in [('s2c',s),('smooth',sm)]:
        ax_err[name].append(np.abs(cl[f'{h}_trans']-g[f'{h}_trans'])[v].mean(0))
        ok=v&(g[f'{h}_trans'][:,2]>1)
        if ok.any(): proj[name].append(np.linalg.norm(P(cl[f'{h}_trans'][ok])-P(g[f'{h}_trans'][ok]),axis=1).mean())
        wrot[name].append(geodesic_deg(aa2m(cl[f'{h}_pose'][v,:3]),aa2m(g[f'{h}_pose'][v,:3])).mean())
keys=sorted(set(k for _,k in rows))
print(f"{'metric (mean over clips x hands)':34s}{'S2C raw':>12s}{'smoothed':>12s}{'GT':>12s}")
for k in keys:
  print(f"{k:34s}"+"".join((f"{np.nanmean(rows[(n,k)]):12.4g}" if (n,k) in rows else f"{'-':>12s}") for n in ['s2c','smooth','gt']))
for n in ax_err: print(f"{n:7s} |trans err| x,y,z (m) {np.mean(ax_err[n],0).round(3)}  image-plane wrist err {np.mean(proj[n]):.1f}px  global-orient err {np.mean(wrot[n]):.1f} deg")
gz=np.array(gz); print('GT z: min',gz[:,0].min().round(2),'max',gz[:,1].max().round(2),'mean within-clip std',gz[:,2].mean().round(3),'| S2C z within-clip std',gz[:,3].mean().round(3))
print('hands with no GT frames:',empty)
