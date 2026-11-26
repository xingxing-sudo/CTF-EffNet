import numpy as np
import librosa
import os
import pandas as pd
from tqdm import tqdm

SR = 16000
WIN_LEN = 4096
C_NUM = 64
C_MIN, C_MAX = -5, 5
N_MELS = 256
FMIN, FMAX = 10, 8000

audio_dir = "./data/ShipsEar"
csv_path = "./data/ShipsEar/shipsEarMeta_90.csv"

df = pd.read_csv(csv_path)
sampled = df.sample(n=10, random_state=42)


BAND_SPLIT = [(0, 31), (31, 32), (32, 64)]  # R, G, B
BAND_NAMES = ["R (negative chirp)", "G (zero chirp)", "B (positive chirp)"]


band_top2_ratios = {name: [] for name in BAND_NAMES}

for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Processing"):
    y, _ = librosa.load(os.path.join(audio_dir, row['filename']), sr=SR)
    hop = WIN_LEN // 2

    for start in range(0, len(y) - WIN_LEN + 1, hop):
        seg = y[start:start + WIN_LEN]
        if len(seg) < WIN_LEN:
            continue
        t = np.arange(WIN_LEN) / SR
        window = np.hanning(WIN_LEN)


        channel_energy = np.zeros(C_NUM)
        mel_filters = librosa.filters.mel(sr=SR, n_fft=WIN_LEN, n_mels=N_MELS, fmin=FMIN, fmax=FMAX)
        
        for c_idx in range(C_NUM):
            c = C_MIN + c_idx * (C_MAX - C_MIN) / (C_NUM - 1)
            dechirped = seg * window * np.exp(-1j * np.pi * c * t**2)
            spec = np.abs(np.fft.rfft(dechirped, n=WIN_LEN))**2
            mel_spec = np.dot(mel_filters, spec)
            channel_energy[c_idx] = np.sum(mel_spec)


        for band_idx, (c_start, c_end) in enumerate(BAND_SPLIT):
            band_energy = channel_energy[c_start:c_end]
            sorted_e = np.sort(band_energy)[::-1]
            total_e = np.sum(sorted_e)
            if total_e > 0 and len(sorted_e) >= 2:
                top2_ratio = np.sum(sorted_e[:2]) / total_e
                band_top2_ratios[BAND_NAMES[band_idx]].append(top2_ratio)
            elif total_e > 0 and len(sorted_e) == 1:

                band_top2_ratios[BAND_NAMES[band_idx]].append(1.0)

# Write results.
print("\n各子带内部Top-2通道的平均能量占比：")
print("-" * 55)
for name in BAND_NAMES:
    ratios = band_top2_ratios[name]
    if len(ratios) > 0:
        mean_ratio = np.mean(ratios) * 100
        std_ratio = np.std(ratios) * 100
        print(f"{name:25s}: {mean_ratio:5.1f}% ± {std_ratio:4.1f}%  (样本数: {len(ratios)})")