# s2c_tools — running pretrained S2C on the VSC cluster

Files:
- `apply_s2c_fixes.sh` — small fixes to S2C's own code (run once in a fresh S2C clone).
- `run_one.py` — runs S2C's `infer.py` on one test clip, without rendering/metrics; saves raw (unsmoothed) output.
- `cache_predictions.py` — runs S2C on N test clips (one per video) and saves `<clip>_s2c.npz` + `<clip>_gt.npz` in the format of `docs/refiner_interface.md`, plus `splits.json` (20/4/4 by video).
- `export_audio.py` — saves `cache/audio/<clip>.wav`: the first 8 s of each cached clip's original audio, aligned with frames 0–239.

## Setup (once)
    cd $VSC_DATA && git clone https://github.com/monkek123King/S2C.git && cd S2C
    bash /path/to/s2c_tools/apply_s2c_fixes.sh
    cp /path/to/s2c_tools/*.py .
    # checkpoints (HF, gated: accept terms on huggingface.co/thuteam/S2C first)
    #   checkpoints/s2c/{piano2pose,piano2posi}  <- thuteam/S2C
    #   checkpoints/hubert-large-ls960-ft        <- facebook/hubert-large-ls960-ft
    # dataset: PianoMotion10M from Zenodo 13297386, unzip with `unzip -DD` on scratch
    #   (otherwise scratch auto-cleanup deletes files with old dates)

## Each GPU session
    srun -A lp_deeplearning_gt_2026b -M wice -p gpu_a100_debug -n 1 --gpus-per-node=1 -t 00:59:00 --pty bash -l
    module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl
    export PYTHONPATH=$HOME/.local/lib/python3.13/site-packages:$PYTHONPATH
    pip install --user "transformers<5" "huggingface-hub>=0.34,<1.0" einops mido librosa opencv-python-headless imageio matplotlib tqdm   # first time only
    cd $VSC_DATA/S2C
    python cache_predictions.py --data_root $VSC_SCRATCH/pianomotion     # ~29 s per clip on A100
    python export_audio.py --data_root $VSC_SCRATCH/pianomotion          # CPU is enough

## Result (14/11)
28 pairs in `cache/` (S2C test split, first 8 s of each clip, raw S2C output).
Baseline, right hand, valid frames: wrist x–y error 3.5 cm; Savitzky-Golay smoothing does not change the error.
