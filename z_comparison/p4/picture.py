# Use a headless plotting backend
import matplotlib
matplotlib.use('Agg')

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import librosa

# Configuration
BASE_DIR = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "z_comparison/p4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PDF = os.path.join(OUTPUT_DIR, "feature_comparison_full_with_axes1.pdf")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "feature_comparison_full_with_axes1.png")
LOG_FILE = os.path.join(OUTPUT_DIR, "load_images_log.txt")

CLASS_MAPPING = {
    "A": "A_Large_Merchant_Ship",
    "B": "B_Small_Speedboat",
    "C": "C_Fishing_Boat",
    "D": "D_Ferry",
    "E": "E_Ambient_Noise"
}

FEATURE_MAPPING = {
    "mel": os.path.join(BASE_DIR, "mel/mel_picture/all_mel_picture"),
    "stft": os.path.join(BASE_DIR, "stft/stft_picture/all_stft_picture"),
    "cqt": os.path.join(BASE_DIR, "cqt/cqt_picture/all_cqt_picture"),
    "wst": os.path.join(BASE_DIR, "wst/wst_picture/all_wst_picture"),
    "chirplet": os.path.join(BASE_DIR, "chirplet/chirplet_picture_pro/all_chirplet_picture")
}

COL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]
FEATURE_TITLES = ["Mel", "STFT", "CQT", "WST", "PTCC-Spec"]


SR = 16000
DURATION = 0.5
WINDOW_SAMPLES = int(DURATION * SR)

# ---------- Mel ----------
MEL_N_FFT = 2048
MEL_HOP = 512
MEL_N_MELS = 128
MEL_FMIN = 50
MEL_FMAX = 8000
MEL_TIME_FRAMES = 1 + (WINDOW_SAMPLES - MEL_N_FFT) // MEL_HOP
MEL_FREQS_HZ = librosa.mel_frequencies(n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)

# ---------- STFT ----------
STFT_N_FFT = 256
STFT_HOP = 512
STFT_N_FREQ = STFT_N_FFT // 2 + 1
STFT_TIME_FRAMES = 1 + (WINDOW_SAMPLES - STFT_N_FFT) // STFT_HOP
STFT_FREQS_HZ = np.linspace(0, SR/2, STFT_N_FREQ)

# ---------- CQT ----------
CQT_FMIN = 50
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 84
CQT_HOP = 512
CQT_TIME_FRAMES = int(np.ceil(WINDOW_SAMPLES / CQT_HOP))
CQT_FREQS_HZ = librosa.cqt_frequencies(n_bins=CQT_N_BINS, fmin=CQT_FMIN, bins_per_octave=CQT_BINS_PER_OCTAVE)

# ---------- WST ----------
WST_N_COEFFS = 169
WST_TIME_FRAMES = 125

# ---------- Chirplet ----------
CHIRP_WIN_LEN = 4096
CHIRP_HOP = 512
CHIRP_N_BINS = 256
CHIRP_FMIN = 10
CHIRP_FMAX = 8000
CHIRP_TIME_FRAMES = 1 + (WINDOW_SAMPLES - CHIRP_WIN_LEN) // CHIRP_HOP


class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
    def write(self, msg):
        self.file.write(msg)
        self.file.flush()
    def flush(self):
        self.file.flush()
    def close(self):
        self.file.close()

original_stdout = sys.stdout
sys.stdout = Tee(LOG_FILE)

# Helper functions
def get_audio_basename(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    if parts[-1].startswith('seg'):
        parts = parts[:-1]
    if parts[-1] in ['mel', 'stft', 'cqt', 'wst', 'chirplet']:
        parts = parts[:-1]
    return '_'.join(parts)

def get_image_files(feat_type, class_full_name):
    feat_dir = FEATURE_MAPPING[feat_type]
    class_dir = os.path.join(feat_dir, class_full_name)
    if not os.path.isdir(class_dir):
        return []
    files = os.listdir(class_dir)
    image_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG')
    image_files = [f for f in files if f.endswith(image_exts)]
    return [os.path.join(class_dir, f) for f in image_files]

def choose_common_basename(class_short):
    class_full = CLASS_MAPPING[class_short]
    basename_counter = Counter()
    feature_files = {}
    for feat_type in FEATURE_MAPPING.keys():
        file_paths = get_image_files(feat_type, class_full)
        basenames = []
        for fp in file_paths:
            fname = os.path.basename(fp)
            bn = get_audio_basename(fname)
            basenames.append(bn)
            basename_counter[bn] += 1
        feature_files[feat_type] = list(zip(file_paths, basenames))
    if not basename_counter:
        return None, {feat: None for feat in FEATURE_MAPPING}, {}
    most_common = basename_counter.most_common(1)[0][0]
    selected = {}
    warnings = {}
    for feat_type in FEATURE_MAPPING.keys():
        matched = None
        for fp, bn in feature_files.get(feat_type, []):
            if bn == most_common:
                matched = fp
                break
        if matched is None:
            file_paths = get_image_files(feat_type, class_full)
            if file_paths:
                matched = file_paths[0]
                warnings[feat_type] = f"未找到基础名 '{most_common}'，改用第一张图片: {os.path.basename(matched)}"
            else:
                warnings[feat_type] = f"特征文件夹为空"
        selected[feat_type] = matched
    return most_common, selected, warnings


def set_axes_ticks(ax, feat_type, img_shape):
    num_ticks = 5
    def map_index_to_pixel(idx, max_idx, max_pixel):
        return int(idx / max_idx * max_pixel)

    if feat_type == 'mel':
        t_idx = np.linspace(0, MEL_TIME_FRAMES-1, num_ticks, dtype=int)
        t_pixel = [map_index_to_pixel(i, MEL_TIME_FRAMES-1, img_shape[1]-1) for i in t_idx]
        t_label = [f"{i/(MEL_TIME_FRAMES-1)*DURATION:.1f}" for i in t_idx]
        ax.set_xticks(t_pixel)
        ax.set_xticklabels(t_label, fontsize=6)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=3)
        f_idx = np.linspace(0, MEL_N_MELS-1, num_ticks, dtype=int)
        f_pixel = [map_index_to_pixel(i, MEL_N_MELS-1, img_shape[0]-1) for i in f_idx]
        f_label = [f"{MEL_FREQS_HZ[i]:.0f}" for i in f_idx]
        ax.set_yticks(f_pixel)
        ax.set_yticklabels(f_label, fontsize=6)
        ax.set_ylabel("Frequency (Hz)", fontsize=7, labelpad=3)

    elif feat_type == 'stft':
        t_idx = np.linspace(0, STFT_TIME_FRAMES-1, num_ticks, dtype=int)
        t_pixel = [map_index_to_pixel(i, STFT_TIME_FRAMES-1, img_shape[1]-1) for i in t_idx]
        t_label = [f"{i/(STFT_TIME_FRAMES-1)*DURATION:.1f}" for i in t_idx]
        ax.set_xticks(t_pixel)
        ax.set_xticklabels(t_label, fontsize=6)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=3)
        f_idx = np.linspace(0, STFT_N_FREQ-1, num_ticks, dtype=int)
        f_pixel = [map_index_to_pixel(i, STFT_N_FREQ-1, img_shape[0]-1) for i in f_idx]
        f_label = [f"{STFT_FREQS_HZ[i]:.0f}" for i in f_idx]
        ax.set_yticks(f_pixel)
        ax.set_yticklabels(f_label, fontsize=6)
        ax.set_ylabel("Frequency (Hz)", fontsize=7, labelpad=3)

    elif feat_type == 'cqt':
        t_idx = np.linspace(0, CQT_TIME_FRAMES-1, num_ticks, dtype=int)
        t_pixel = [map_index_to_pixel(i, CQT_TIME_FRAMES-1, img_shape[1]-1) for i in t_idx]
        t_label = [f"{i/(CQT_TIME_FRAMES-1)*DURATION:.1f}" for i in t_idx]
        ax.set_xticks(t_pixel)
        ax.set_xticklabels(t_label, fontsize=6)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=3)
        f_idx = np.linspace(0, CQT_N_BINS-1, num_ticks, dtype=int)
        f_pixel = [map_index_to_pixel(i, CQT_N_BINS-1, img_shape[0]-1) for i in f_idx]
        f_label = [f"{CQT_FREQS_HZ[i]:.0f}" for i in f_idx]
        ax.set_yticks(f_pixel)
        ax.set_yticklabels(f_label, fontsize=6)
        ax.set_ylabel("Frequency (Hz)", fontsize=7, labelpad=3)

    elif feat_type == 'wst':
        t_idx = np.linspace(0, WST_TIME_FRAMES-1, num_ticks, dtype=int)
        t_pixel = [map_index_to_pixel(i, WST_TIME_FRAMES-1, img_shape[1]-1) for i in t_idx]
        t_label = [f"{i/(WST_TIME_FRAMES-1)*DURATION:.1f}" for i in t_idx]
        ax.set_xticks(t_pixel)
        ax.set_xticklabels(t_label, fontsize=6)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=3)
        s_idx = np.linspace(WST_N_COEFFS-1, 0, num_ticks, dtype=int)
        s_pixel = [map_index_to_pixel(WST_N_COEFFS-1 - i, WST_N_COEFFS-1, img_shape[0]-1) for i in s_idx]
        s_label = [f"{i}" for i in s_idx]
        ax.set_yticks(s_pixel)
        ax.set_yticklabels(s_label, fontsize=6)
        ax.set_ylabel("Scattering path index", fontsize=7, labelpad=3)

    elif feat_type == 'chirplet':
        t_idx = np.linspace(0, CHIRP_TIME_FRAMES-1, num_ticks, dtype=int)
        t_pixel = [map_index_to_pixel(i, CHIRP_TIME_FRAMES-1, img_shape[1]-1) for i in t_idx]
        t_label = [f"{i/(CHIRP_TIME_FRAMES-1)*DURATION:.1f}" for i in t_idx]
        ax.set_xticks(t_pixel)
        ax.set_xticklabels(t_label, fontsize=6)
        ax.set_xlabel("Time (s)", fontsize=7, labelpad=3)
        f_idx = np.linspace(0, CHIRP_N_BINS-1, num_ticks, dtype=int)
        f_pixel = [map_index_to_pixel(i, CHIRP_N_BINS-1, img_shape[0]-1) for i in f_idx]
        chirp_freqs = librosa.mel_frequencies(n_mels=CHIRP_N_BINS, fmin=CHIRP_FMIN, fmax=CHIRP_FMAX)
        f_label = [f"{chirp_freqs[i]:.0f}" for i in f_idx]
        ax.set_yticks(f_pixel)
        ax.set_yticklabels(f_label, fontsize=6)
        ax.set_ylabel("Frequency (Hz)", fontsize=7, labelpad=3)

    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

    ax.tick_params(axis='both', which='major', labelsize=6)


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(14, 12), dpi=300)
cmap = 'magma'

short_classes = list(CLASS_MAPPING.keys())
feature_types = list(FEATURE_MAPPING.keys())

print("=" * 60)
print("正在加载以下图片（少数服从多数原则，方向统一为 origin='lower'）：")
print("-" * 60)

class_selection = {}
for class_short in short_classes:
    best_bn, selected, warns = choose_common_basename(class_short)
    class_selection[class_short] = (best_bn, selected, warns)
    if best_bn is None:
        print(f"⚠️ 类别 {class_short}: 未找到任何图片文件")
    else:
        print(f"类别 {class_short}: 选中基础名 = {best_bn}")
        for ft, warn in warns.items():
            if warn:
                print(f"  [警告] {ft.upper()}: {warn}")

print("=" * 60)


im = None
for i, class_short in enumerate(short_classes):
    best_bn, selected, warns = class_selection[class_short]
    for j, feat_type in enumerate(feature_types):
        ax = axes[i, j]
        img_path = selected.get(feat_type)
        if img_path is None or not os.path.exists(img_path):
            ax.imshow(np.zeros((224, 224)), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{FEATURE_TITLES[j]}", fontsize=10, pad=8)
            continue

        if feat_type == 'chirplet':
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img).astype(np.float32)
            


            fused = 0.6 * img_array[:, :, 0] + 0.3 * img_array[:, :, 1] + 0.1 * img_array[:, :, 2]
            

            p2, p98 = np.percentile(fused, (2, 98))
            fused = np.clip((fused - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
            

            im = ax.imshow(fused, cmap=cmap, aspect='auto', origin='upper', vmin=0, vmax=255)
        else:
            img = Image.open(img_path).convert("L")
            img_array = np.array(img)
            im = ax.imshow(img_array, cmap=cmap, aspect='auto', origin='upper', vmin=0, vmax=255)

        set_axes_ticks(ax, feat_type, img_array.shape)
        if i == 0:
            ax.set_title(f"{FEATURE_TITLES[j]}", fontsize=10, pad=8)


for i, class_short in enumerate(short_classes):
    ax0 = axes[i, 0]
    pos = ax0.get_position()
    fig.text(pos.x0 - 0.045, pos.y0 + pos.height/2,
             f"Class {class_short}",
             ha='right', va='center', fontsize=9, rotation=90)

print("=" * 60)


for j, label in enumerate(COL_LABELS):
    ax_bottom = axes[4, j]
    pos = ax_bottom.get_position()
    fig.text(pos.x0 + pos.width/2, pos.y0 - 0.035,
             label, ha='center', va='top', fontsize=10)



top_ax = axes[0, 0].get_position().y1
bottom_ax = axes[-1, 0].get_position().y0
cbar_height = top_ax - bottom_ax
cbar_bottom = bottom_ax
cbar_ax = fig.add_axes([0.92, cbar_bottom, 0.02, cbar_height])
fig.colorbar(im, cax=cbar_ax)


plt.subplots_adjust(left=0.12, wspace=0.35, hspace=0.35, right=0.9)

plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
plt.savefig(OUTPUT_PNG, format='png', bbox_inches='tight')
plt.close()

print("✅ 5×5 特征对比图已生成完成！")
print(f"📄 PDF：{OUTPUT_PDF}")
print(f"🖼️ PNG：{OUTPUT_PNG}")
print(f"📝 日志：{LOG_FILE}")

sys.stdout.close()
sys.stdout = original_stdout
