# S2C source in the piano project

This directory contains the official S2C source, imported without code changes.
`UPSTREAM.json` pins the source commit. The original README and MIT LICENSE are
preserved. Files are ordinary files in the parent repository, not a submodule;
future model changes can be reviewed alongside the rest of this project.

## Entry points

| File | Role |
| --- | --- |
| `train.py` | Train the hand-position predictors |
| `train_diffusion.py` | Train the dual-stream pose diffusion model |
| `infer.py` | Run pretrained motion generation |
| `models/piano2posi.py` | Audio encoder and position predictor |
| `models/denoise_diffusion.py` | Temporal U-Net, diffusion sampling and losses |
| `models/diff_multiAttention.py` | Cross-hand fusion / HCAA implementation |
| `datasets/PianoPose.py` | Dataset loading and motion representation |

Run upstream scripts with this directory as the working directory; they use
relative paths. See the upstream README for dependencies and training arguments.

## Existing project tools

From a Linux cluster checkout of the parent project:

```bash
cd /path/to/deep_learning/third_party/S2C
# Optional: apply the team's existing inference compatibility fixes.
bash ../../s2c_tools/apply_s2c_fixes.sh
cp ../../s2c_tools/*.py .
```

The import itself does NOT apply that patch, install dependencies, download model
weights or datasets, or start training. The patch changes tracked upstream files;
review its diff before running experiments. The existing `s2c_tools/README.md`
describes the earlier standalone checkout; use this directory instead of
`$VSC_DATA/S2C` when working with the integrated source.

S2C, HuBERT and MANO assets and PianoMotion10M data must be obtained separately.
The upstream `.gitignore` excludes its data, checkpoint and output directories.
This import has only been checked for source integrity and Python syntax; it does
not establish that GPU inference or training runs in the current environment.
