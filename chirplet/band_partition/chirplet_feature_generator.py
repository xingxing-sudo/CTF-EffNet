
import matplotlib
matplotlib.use('Agg')
import os, shutil, sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from PIL import Image, ImageOps
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.3
IMAGE_SIZE_PIXEL = (224, 224)
KEEP_ASPECT_RATIO = False

AUDIO_ROOT_DIR = "./data/ShipsEar"
CSV_FILE_PATH = "./data/ShipsEar/shipsEarMeta_90.csv"
OUTPUT_BASE_DIR = "./chirplet/band_partition/chirplet_features"

CLASS_MAPPING = {
    "Ocean liner": "A_Large_Merchant_Ship", "RORO": "A_Large_Merchant_Ship", "Dredger": "A_Large_Merchant_Ship",
    "Motorboat": "B_Small_Speedboat", "Tugboat": "B_Small_Speedboat", "Pilot ship": "B_Small_Speedboat",
    "Sailboat": "B_Small_Speedboat", "Yacht": "B_Small_Speedboat", "High speed motorboat": "B_Small_Speedboat",
    "Zodiac": "B_Small_Speedboat", "Fishboat": "C_Fishing_Boat", "Trawler": "C_Fishing_Boat",
    "Mussel boat": "C_Fishing_Boat", "fishboat": "C_Fishing_Boat", "Passengers": "D_Ferry",
    "Natural ambient noise": "E_Ambient_Noise"
}

# Chirplet parameters
CHIRP_WIN_LENGTH = 4096
CHIRP_HOP_LENGTH = 512
CHIRP_N_BINS = 256
CHIRP_FMIN = 10
CHIRP_FMAX = 8000
CHIRP_TOP_DB = 80.0
CHIRP_C_RANGE = (-5, 5, 64)
CHIRP_AGG_TOPK = 5
USE_MEL_FREQ_AXIS = True
USE_SPECTRAL_ENHANCE = True
ENHANCE_GAMMA = 0.6
USE_ADAPTIVE_NORMALIZATION = True
NORMAL_PERCENTILE = 99.5
RESIZE_MODE = Image.Resampling.LANCZOS
USE_GLOBAL_TOPK = True

# Test-mode flag
TEST_MODE = False
TEST_FILES_PER_CLASS = 2  # Recordings per class in test mode

# Define the band partitions.
# Each scheme specifies the exclusive upper boundary for R and G.
# R: 0:R_end, G: R_end:G_end, B: G_end:64
BAND_SCHEMES = [
    ("band_1bin",  31, 32),
    ("band_3bin",  30, 33),   # G:30-32
    ("band_5bin",  29, 34),   # G:29-33
    ("band_22bin", 21, 43),  # Earlier approximately equal-width bands
]

# Feature generator
class ChirpletFeatureGenerator:
    def __init__(self, sr, band_split):
        self.sr = sr
        self.band_split = band_split
        self.win_length = CHIRP_WIN_LENGTH
        self.hop_length = CHIRP_HOP_LENGTH
        self.n_bins = CHIRP_N_BINS
        self.fmin = CHIRP_FMIN
        self.fmax = min(CHIRP_FMAX, sr // 2)
        self.top_db = CHIRP_TOP_DB
        self.agg_topk = CHIRP_AGG_TOPK
        self.c_min, self.c_max, self.c_num = CHIRP_C_RANGE
        self.c_values = np.linspace(self.c_min, self.c_max, self.c_num, dtype=np.float32)
        self.t = (np.arange(self.win_length, dtype=np.float32) - self.win_length // 2) / self.sr
        self.window = np.hanning(self.win_length).astype(np.float32)
        self.n_fft = self.win_length
        self.full_fft_freqs = fftfreq(self.n_fft, 1 / self.sr).astype(np.float32)
        self.positive_freq_idx = self.full_fft_freqs >= 0
        self.fft_freqs = self.full_fft_freqs[self.positive_freq_idx]
        if USE_MEL_FREQ_AXIS:
            self.target_freqs = librosa.mel_frequencies(n_mels=self.n_bins, fmin=self.fmin, fmax=self.fmax).astype(np.float32)
        else:
            self.target_freqs = np.linspace(self.fmin, self.fmax, self.n_bins, dtype=np.float32)
        self.dechirp_phase = np.exp(-1j * np.pi * self.c_values[:, np.newaxis] * (self.t ** 2)).astype(np.complex64)

    def _compute_chirplet_cube(self, audio):
        n_samples = len(audio)
        n_frames = (n_samples - self.win_length) // self.hop_length + 1
        if n_frames <= 0:
            return np.zeros((self.c_num, self.n_bins, 1), dtype=np.float32)
        chirp_spec_cube = np.zeros((self.c_num, self.n_bins, n_frames), dtype=np.float32)
        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_length
            frame = audio[start:start+self.win_length].astype(np.float32)
            frame_windowed = frame * self.window
            dechirped_frames = frame_windowed[np.newaxis, :] * self.dechirp_phase
            fft_results = fft(dechirped_frames, n=self.n_fft, axis=-1)
            mag_spectrum = np.abs(fft_results) ** 2
            for c_idx in range(self.c_num):
                pos_mag = mag_spectrum[c_idx, self.positive_freq_idx]
                mag_interp = np.interp(self.target_freqs, self.fft_freqs, pos_mag)
                chirp_spec_cube[c_idx, :, frame_idx] = mag_interp
        return chirp_spec_cube

    def _aggregate_band(self, band_cube):
        if USE_GLOBAL_TOPK:
            band_energy = np.sum(band_cube, axis=(1,2))
            n_top = min(self.agg_topk//2, band_cube.shape[0])
            if n_top == 0: n_top = 1
            top_idx = np.argsort(band_energy)[::-1][:n_top]
            top_cube = band_cube[top_idx, :, :]
            top_energy = band_energy[top_idx]
            weights = top_energy / (np.sum(top_energy) + 1e-10)
            weights = weights[:, np.newaxis, np.newaxis]
            agg = np.sum(top_cube * weights, axis=0)
        else:
            sorted_cube = np.sort(band_cube, axis=0)[::-1]
            n_top = min(self.agg_topk//2, band_cube.shape[0])
            if n_top == 0: n_top = 1
            top_cube = sorted_cube[:n_top]
            weights = top_cube / (np.sum(top_cube, axis=0, keepdims=True) + 1e-10)
            agg = np.sum(top_cube * weights, axis=0)
        return agg

    def _normalize_spec(self, spec_db):
        if len(spec_db.shape) == 3:
            rgb_spec = np.zeros_like(spec_db, dtype=np.float32)
            for ch in range(3):
                channel = spec_db[:,:,ch]
                max_val = np.percentile(channel, NORMAL_PERCENTILE)
                min_val = np.min(channel)
                channel_clipped = np.clip(channel, min_val, max_val)
                channel_norm = (channel_clipped - min_val) / (max_val - min_val + 1e-10)
                if USE_SPECTRAL_ENHANCE:
                    channel_norm = np.power(channel_norm, ENHANCE_GAMMA)
                rgb_spec[:,:,ch] = channel_norm
            return (rgb_spec * 255).astype(np.uint8)
        else:
            max_val = np.percentile(spec_db, NORMAL_PERCENTILE)
            min_val = np.min(spec_db)
            spec_clipped = np.clip(spec_db, min_val, max_val)
            spec_norm = (spec_clipped - min_val) / (max_val - min_val + 1e-10)
            if USE_SPECTRAL_ENHANCE:
                spec_norm = np.power(spec_norm, ENHANCE_GAMMA)
            return (spec_norm * 255).astype(np.uint8)

    def compute_feature(self, audio):
        cube = self._compute_chirplet_cube(audio)
        rgb = np.zeros((self.n_bins, cube.shape[2], 3), dtype=np.uint8)
        for ch, (c_start, c_end) in enumerate(self.band_split):
            band_cube = cube[c_start:c_end, :, :]
            band_agg = self._aggregate_band(band_cube)
            band_agg = band_agg + 1e-10
            band_db = librosa.power_to_db(band_agg, ref=np.max, top_db=self.top_db)
            rgb[:,:,ch] = self._normalize_spec(band_db)
        return rgb

def save_feature(feat, save_path):
    img = Image.fromarray(feat, mode='RGB')
    img = img.resize(IMAGE_SIZE_PIXEL, RESIZE_MODE)
    img.save(save_path)

# Audio processing
class AudioProcessor:
    def __init__(self, feature_gen):
        self.feature_gen = feature_gen
    def process_audio(self, audio_path, output_dir, audio_basename):
        y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        y = y.astype(np.float32)
        win_samples = int(WINDOW_DURATION * TARGET_SR)
        step_samples = win_samples - int(OVERLAP_DURATION * TARGET_SR)
        start = 0
        seg_idx = 0
        while start + win_samples <= len(y):
            seg = y[start:start+win_samples]
            feat = self.feature_gen.compute_feature(seg)
            fname = f"{audio_basename}_chirplet_seg{seg_idx+1:03d}.png"
            save_feature(feat, os.path.join(output_dir, fname))
            start += step_samples
            seg_idx += 1
        return seg_idx


def generate_for_scheme(scheme_name, band_split, output_root):
    print(f"\n{'='*60}\n生成方案：{scheme_name}，分带：{band_split}\n{'='*60}")
    unique_classes = sorted(set(CLASS_MAPPING.values()))
    out_dir = os.path.join(output_root, scheme_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for cls in unique_classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    df = pd.read_csv(CSV_FILE_PATH)
    df['mapped_class'] = df['type'].map(CLASS_MAPPING)
    df = df.dropna(subset=['mapped_class'])

    feature_gen = ChirpletFeatureGenerator(TARGET_SR, band_split)
    processor = AudioProcessor(feature_gen)

    for cls in unique_classes:
        cls_df = df[df['mapped_class'] == cls]
        if TEST_MODE:
            cls_df = cls_df.head(TEST_FILES_PER_CLASS)
            print(f"  [TEST MODE] 类别 {cls} 仅处理前 {len(cls_df)} 条音频")
        else:
            print(f"处理类别：{cls} ({len(cls_df)} 条音频)")
        for _, row in tqdm(cls_df.iterrows(), total=len(cls_df), desc=f"  {cls}"):
            fname = row['filename']
            apath = os.path.join(AUDIO_ROOT_DIR, fname)
            if not os.path.exists(apath):
                continue
            basename = os.path.splitext(fname)[0]
            processor.process_audio(apath, os.path.join(out_dir, cls), basename)
    print(f"方案 {scheme_name} 完成。")

def generate_nosplit_scheme(output_root):
    scheme_name = "band_nosplit"
    print(f"\n{'='*60}\n生成方案：{scheme_name} (无划分，单通道)\n{'='*60}")
    unique_classes = sorted(set(CLASS_MAPPING.values()))
    out_dir = os.path.join(output_root, scheme_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for cls in unique_classes:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    df = pd.read_csv(CSV_FILE_PATH)
    df['mapped_class'] = df['type'].map(CLASS_MAPPING)
    df = df.dropna(subset=['mapped_class'])

    class NoSplitGen:
        def __init__(self, sr):
            self.sr = sr
            self.win_length = CHIRP_WIN_LENGTH
            self.hop_length = CHIRP_HOP_LENGTH
            self.n_bins = CHIRP_N_BINS
            self.fmin = CHIRP_FMIN
            self.fmax = min(CHIRP_FMAX, sr//2)
            self.top_db = CHIRP_TOP_DB
            self.c_min, self.c_max, self.c_num = CHIRP_C_RANGE
            self.c_values = np.linspace(self.c_min, self.c_max, self.c_num, dtype=np.float32)
            self.t = (np.arange(self.win_length, dtype=np.float32)-self.win_length//2)/sr
            self.window = np.hanning(self.win_length).astype(np.float32)
            self.n_fft = self.win_length
            self.full_fft_freqs = fftfreq(self.n_fft, 1/sr).astype(np.float32)
            self.positive_freq_idx = self.full_fft_freqs>=0
            self.fft_freqs = self.full_fft_freqs[self.positive_freq_idx]
            if USE_MEL_FREQ_AXIS:
                self.target_freqs = librosa.mel_frequencies(n_mels=self.n_bins, fmin=self.fmin, fmax=self.fmax).astype(np.float32)
            else:
                self.target_freqs = np.linspace(self.fmin, self.fmax, self.n_bins, dtype=np.float32)
            self.dechirp_phase = np.exp(-1j*np.pi*self.c_values[:,np.newaxis]*(self.t**2)).astype(np.complex64)

        def compute(self, audio):
            n_samples = len(audio)
            n_frames = (n_samples - self.win_length)//self.hop_length + 1
            if n_frames<=0:
                return np.zeros((self.n_bins, 1, 3), dtype=np.uint8)
            chirp_cube = np.zeros((self.c_num, self.n_bins, n_frames), dtype=np.float32)
            for frame_idx in range(n_frames):
                start = frame_idx*self.hop_length
                frame = audio[start:start+self.win_length].astype(np.float32)
                frame_windowed = frame * self.window
                dechirped = frame_windowed[np.newaxis,:]*self.dechirp_phase
                fft_res = fft(dechirped, n=self.n_fft, axis=-1)
                mag = np.abs(fft_res)**2
                for c_idx in range(self.c_num):
                    pos_mag = mag[c_idx, self.positive_freq_idx]
                    interp = np.interp(self.target_freqs, self.fft_freqs, pos_mag)
                    chirp_cube[c_idx, :, frame_idx] = interp

            total_energy = np.sum(chirp_cube, axis=(1,2))
            top2_idx = np.argsort(total_energy)[::-1][:2]
            top_energy = total_energy[top2_idx]
            weights = top_energy / (np.sum(top_energy)+1e-10)
            weights = weights[:, np.newaxis, np.newaxis]
            agg_spec = np.sum(chirp_cube[top2_idx]*weights, axis=0)
            agg_spec = agg_spec + 1e-10
            agg_db = librosa.power_to_db(agg_spec, ref=np.max, top_db=self.top_db)
            max_val = np.percentile(agg_db, NORMAL_PERCENTILE)
            min_val = np.min(agg_db)
            clipped = np.clip(agg_db, min_val, max_val)
            norm = (clipped - min_val) / (max_val - min_val + 1e-10)
            if USE_SPECTRAL_ENHANCE:
                norm = np.power(norm, ENHANCE_GAMMA)
            gray = (norm*255).astype(np.uint8)
            rgb = np.stack([gray, gray, gray], axis=-1)
            return rgb

    gen = NoSplitGen(TARGET_SR)
    for cls in unique_classes:
        cls_df = df[df['mapped_class'] == cls]
        if TEST_MODE:
            cls_df = cls_df.head(TEST_FILES_PER_CLASS)
            print(f"  [TEST MODE] 类别 {cls} 仅处理前 {len(cls_df)} 条音频")
        else:
            print(f"处理类别：{cls} ({len(cls_df)} 条音频)")
        for _, row in tqdm(cls_df.iterrows(), total=len(cls_df), desc=f"  {cls}"):
            fname = row['filename']
            apath = os.path.join(AUDIO_ROOT_DIR, fname)
            if not os.path.exists(apath):
                continue
            basename = os.path.splitext(fname)[0]
            y, _ = librosa.load(apath, sr=TARGET_SR, mono=True)
            y = y.astype(np.float32)
            win_samples = int(WINDOW_DURATION*TARGET_SR)
            step_samples = win_samples - int(OVERLAP_DURATION*TARGET_SR)
            start = 0
            seg_idx = 0
            while start + win_samples <= len(y):
                seg = y[start:start+win_samples]
                feat = gen.compute(seg)
                save_feature(feat, os.path.join(out_dir, cls, f"{basename}_chirplet_seg{seg_idx+1:03d}.png"))
                start += step_samples
                seg_idx += 1
    print(f"方案 {scheme_name} 完成。")


if __name__ == "__main__":
    print("=" * 70)
    if TEST_MODE:
        print(f"⚠️  测试模式启用：每类仅处理 {TEST_FILES_PER_CLASS} 条音频")
    else:
        print("🔵 全量生成模式：处理全部 90 条音频")
    print("=" * 70)

    # Generate all band-partitioned representations.
    for name, r_end, g_end in BAND_SCHEMES:
        band_split = [(0, r_end), (r_end, g_end), (g_end, 64)]
        generate_for_scheme(name, band_split, OUTPUT_BASE_DIR)

    # Generate the unpartitioned representation.
    generate_nosplit_scheme(OUTPUT_BASE_DIR)

    print("\n全部方案特征生成完毕！")