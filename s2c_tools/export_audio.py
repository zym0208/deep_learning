# Save the audio for each cached clip: first 8 s of the original mp3 (same window as frames 0-239 in the .npz).
# Usage (in $VSC_DATA/S2C, after cache_predictions.py): python export_audio.py --data_root $VSC_SCRATCH/pianomotion
import argparse, json, os, librosa, soundfile as sf
p = argparse.ArgumentParser()
p.add_argument('--data_root', required=True)
p.add_argument('--cache', default='cache')
p.add_argument('--seconds', type=int, default=8)
a = p.parse_args()
s = json.load(open(f'{a.cache}/splits.json'))
os.makedirs(f'{a.cache}/audio', exist_ok=True)
for split in ['train', 'val', 'test']:
    for c in s[split]:
        y, sr = librosa.load(f"{a.data_root}/audio/{c['up']}/{c['video']}/{c['clip']}.mp3", sr=None)
        y = y[:a.seconds * sr]
        sf.write(f"{a.cache}/audio/{c['clip']}.wav", y, sr, subtype='PCM_16')
        print(split, c['clip'], f'{len(y)/sr:.2f} s', sr, 'Hz', flush=True)
print('done')
