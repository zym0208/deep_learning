#!/bin/bash
# Changes we made to a fresh clone of https://github.com/monkek123King/S2C
# so that infer.py runs on the VSC wICE cluster (PyTorch 2.9.1 module).
# Usage: cd S2C && bash ../s2c_tools/apply_s2c_fixes.sh
set -e
sed -i '/CUDA_VISIBLE_DEVICES/d' infer.py                      # don't override the Slurm GPU
sed -i "s/^    model_path_left = sorted.*/    model_path_left = 'piano2pose-left.ckpt'/; s/^    model_path_right = sorted.*/    model_path_right = 'piano2pose-right.ckpt'/" infer.py   # HF checkpoint names
sed -i "s/args.up_list = \[.*\]/args.up_list = []/" infer.py   # use all pianists, not 2 hard-coded ones
sed -i 's/HubertModel.from_pretrained(args.wav2vec_path)/HubertModel.from_pretrained(args.wav2vec_path, attn_implementation="eager")/' models/piano2posi.py   # newer transformers
touch datasets/__init__.py models/__init__.py                  # avoid clash with the HF "datasets" package
echo "S2C fixes applied"
