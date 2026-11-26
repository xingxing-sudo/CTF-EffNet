# Use a headless plotting backend
import matplotlib
matplotlib.use('Agg')

import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
from scipy.fft import fft, fftfreq
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

warnings.filterwarnings('ignore')

# ==============================================================================

# ==============================================================================
# Dataset paths
AUDIO_ROOT_DIR = "./deepship/data/Deepship"
OUTPUT_ROOT_DIR = "./deepship/chirplet/chirplet_picture/all_chirplet_picture"


RUN_MODE = 'full'


TEST_AUDIO_PATH = "./deepship/data/Deepship/Cargo/1.wav"
TEST_OUTPUT_DIR = "./deepship/chirplet/chirplet_test_picture"
SAVE_CHANNEL_SPLITS = True
RUN_TEST_FIRST = True

TEST_USE_COLORMAP = True

# Audio segmentation
TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.0

# Image output
IMAGE_SIZE_PIXEL = (224, 224)
KEEP_ASPECT_RATIO = False


CLASS_MAPPING = {
    "Cargo": "A_Cargo_Ship",
    "Passengership": "B_Passenger_Ship",
    "Tanker": "C_Tanker_Ship",
    "Tug": "D_Tug_Ship"
}


MAX_FILES_PER_CLASS = 15
SAMPLING_RANDOM_SEED = 42  # Set the random seed

# ==============================================================================

# ==============================================================================
CHIRP_WIN_LENGTH = 4096
CHIRP_HOP_LENGTH = 512
CHIRP_N_BINS = 256
CHIRP_FMIN = 10
CHIRP_FMAX = 8000
CHIRP_TOP_DB = 80.0

CHIRP_C_RANGE = (-5, 5, 64)
CHIRP_AGG_TOPK = 5

USE_RGB_THREE_CHANNEL = True
RGB_CHANNEL_MODE = 'chirplet_band'
USE_MEL_FREQ_AXIS = True
USE_SPECTRAL_ENHANCE = True
ENHANCE_GAMMA = 0.6
USE_ADAPTIVE_NORMALIZATION = True
NORMAL_PERCENTILE = 99.5
RESIZE_MODE = Image.Resampling.LANCZOS

CHIRP_BAND_SPLIT = [
    (0, 21),
    (21, 43),
    (43, 64)
]

USE_GLOBAL_TOPK = True
USE_GLOBAL_PRE_STRETCH = True

# ==============================================================================

# ==============================================================================
class ChirpletFeatureGenerator:
    def __init__(self, sr):
        self.sr = sr
        self.use_rgb = USE_RGB_THREE_CHANNEL
        self.rgb_mode = RGB_CHANNEL_MODE

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
            self.target_freqs = librosa.mel_frequencies(
                n_mels=self.n_bins, fmin=self.fmin, fmax=self.fmax
            ).astype(np.float32)
        else:
            self.target_freqs = np.linspace(self.fmin, self.fmax, self.n_bins, dtype=np.float32)

        self.dechirp_phase = np.exp(-1j * np.pi * self.c_values[:, np.newaxis] * (self.t ** 2)).astype(np.complex64)

    def compute_feature(self, audio):
        if self.use_rgb and self.rgb_mode == 'chirplet_band':
            return self._generate_chirplet_band_rgb(audio)
        else:
            base_spec = self._compute_chirplet(audio)
            if self.use_rgb:
                return self._generate_colormap_rgb(base_spec)
            else:
                return base_spec

    def _compute_chirplet(self, audio, return_full_cube=False):
        n_samples = len(audio)
        n_frames = (n_samples - self.win_length) // self.hop_length + 1
        if n_frames <= 0:
            if return_full_cube:
                return np.zeros((self.c_num, self.n_bins, 1), dtype=np.float32)
            return np.zeros((self.n_bins, 1), dtype=np.uint8)

        chirp_spec_cube = np.zeros((self.c_num, self.n_bins, n_frames), dtype=np.float32)

        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_length
            frame = audio[start:start + self.win_length].astype(np.float32)
            frame_windowed = frame * self.window

            dechirped_frames = frame_windowed[np.newaxis, :] * self.dechirp_phase
            fft_results = fft(dechirped_frames, n=self.n_fft, axis=-1)
            mag_spectrum = np.abs(fft_results) ** 2

            for c_idx in range(self.c_num):
                pos_mag = mag_spectrum[c_idx, self.positive_freq_idx]
                mag_interp = np.interp(self.target_freqs, self.fft_freqs, pos_mag)
                chirp_spec_cube[c_idx, :, frame_idx] = mag_interp

        if return_full_cube:
            return chirp_spec_cube

        if USE_GLOBAL_TOPK:
            total_energy = np.sum(chirp_spec_cube, axis=(1, 2))
            topk_indices = np.argsort(total_energy)[::-1][:self.agg_topk]
            topk_cube = chirp_spec_cube[topk_indices, :, :]
            topk_energy = total_energy[topk_indices]
            epsilon = 1e-10
            weights = topk_energy / (np.sum(topk_energy) + epsilon)
            weights = weights[:, np.newaxis, np.newaxis]
            chirp_spec_agg = np.sum(topk_cube * weights, axis=0)
        else:
            chirp_spec_sorted = np.sort(chirp_spec_cube, axis=0)[::-1]
            topk_spec = chirp_spec_sorted[:self.agg_topk]
            epsilon = 1e-10
            weights = topk_spec / (np.sum(topk_spec, axis=0, keepdim=True) + epsilon)
            chirp_spec_agg = np.sum(topk_spec * weights, axis=0)

        chirp_spec_agg = chirp_spec_agg + 1e-10
        chirp_db = librosa.power_to_db(chirp_spec_agg, ref=np.max, top_db=self.top_db)
        return self._normalize_spec(chirp_db)

    def _normalize_spec(self, spec_db):
        if len(spec_db.shape) == 3:
            if USE_GLOBAL_PRE_STRETCH:
                global_max = np.percentile(spec_db, 99.9)
                global_min = np.min(spec_db)
                spec_db = np.clip(spec_db, global_min, global_max)
                spec_db = (spec_db - global_min) / (global_max - global_min + 1e-10) * self.top_db

            rgb_spec = np.zeros_like(spec_db, dtype=np.float32)
            for channel_idx in range(3):
                channel = spec_db[:, :, channel_idx]

                if USE_ADAPTIVE_NORMALIZATION:
                    max_val = np.percentile(channel, NORMAL_PERCENTILE)
                    min_val = np.min(channel)
                else:
                    max_val = np.max(channel)
                    min_val = np.min(channel)

                channel_clipped = np.clip(channel, min_val, max_val)
                channel_norm = (channel_clipped - min_val) / (max_val - min_val + 1e-10)

                if USE_SPECTRAL_ENHANCE:
                    channel_norm = np.power(channel_norm, ENHANCE_GAMMA)

                rgb_spec[:, :, channel_idx] = channel_norm

            return (rgb_spec * 255).astype(np.uint8)

        if USE_ADAPTIVE_NORMALIZATION:
            max_val = np.percentile(spec_db, NORMAL_PERCENTILE)
            min_val = np.min(spec_db)
        else:
            max_val = np.max(spec_db)
            min_val = np.min(spec_db)

        spec_clipped = np.clip(spec_db, min_val, max_val)
        spec_norm = (spec_clipped - min_val) / (max_val - min_val + 1e-10)

        if USE_SPECTRAL_ENHANCE:
            spec_norm = np.power(spec_norm, ENHANCE_GAMMA)

        return (spec_norm * 255).astype(np.uint8)

    def _generate_chirplet_band_rgb(self, audio):
        chirp_cube = self._compute_chirplet(audio, return_full_cube=True)
        rgb_spec = np.zeros((self.n_bins, chirp_cube.shape[2], 3), dtype=np.uint8)

        for channel_idx, (c_start, c_end) in enumerate(CHIRP_BAND_SPLIT):
            band_cube = chirp_cube[c_start:c_end, :, :]

            if USE_GLOBAL_TOPK:
                band_energy = np.sum(band_cube, axis=(1, 2))
                topk_idx_in_band = np.argsort(band_energy)[::-1][:max(1, self.agg_topk // 2)]
                topk_band = band_cube[topk_idx_in_band, :, :]
                topk_weights = band_energy[topk_idx_in_band] / (np.sum(band_energy[topk_idx_in_band]) + 1e-10)
                topk_weights = topk_weights[:, np.newaxis, np.newaxis]
                band_agg = np.sum(topk_band * topk_weights, axis=0)
            else:
                band_sorted = np.sort(band_cube, axis=0)[::-1]
                topk_band = band_sorted[:max(1, self.agg_topk // 2)]
                weights = topk_band / (np.sum(topk_band, axis=0, keepdim=True) + 1e-10)
                band_agg = np.sum(topk_band * weights, axis=0)

            band_agg = band_agg + 1e-10
            band_db = librosa.power_to_db(band_agg, ref=np.max, top_db=self.top_db)
            rgb_spec[:, :, channel_idx] = self._normalize_spec(band_db)

        return rgb_spec

    def _generate_colormap_rgb(self, gray_spec):
        spec_norm = gray_spec / 255.0
        cmap = cm.get_cmap('viridis')
        rgba_spec = cmap(spec_norm)
        rgb_spec = (rgba_spec[:, :, :3] * 255).astype(np.uint8)
        return rgb_spec


def save_pure_feature(feature_norm, save_path):
    feature_norm = np.transpose(feature_norm, (1, 0, 2)) if len(feature_norm.shape) == 3 else feature_norm.T
    target_w, target_h = IMAGE_SIZE_PIXEL
    is_rgb = len(feature_norm.shape) == 3
    mode = 'RGB' if is_rgb else 'L'

    if KEEP_ASPECT_RATIO:
        time_len = feature_norm.shape[0]
        freq_len = feature_norm.shape[1]
        scale = target_h / freq_len
        new_time_len = int(time_len * scale)
        img = Image.fromarray(feature_norm, mode=mode)
        img = img.resize((new_time_len, target_h), RESIZE_MODE)
        if new_time_len < target_w:
            pad_total = target_w - new_time_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img = ImageOps.expand(img, (pad_left, 0, pad_right, 0), fill=0)
        else:
            crop_left = (new_time_len - target_w) // 2
            img = img.crop((crop_left, 0, crop_left + target_w, target_h))
    else:
        img = Image.fromarray(feature_norm, mode=mode)
        img = img.resize((target_w, target_h), RESIZE_MODE)

    img.save(save_path)


def apply_colormap_to_physical_rgb(physical_rgb, save_path, colormap_names=('Reds', 'Greens', 'Blues')):
    physical_rgb = np.transpose(physical_rgb, (1, 0, 2))
    target_w, target_h = IMAGE_SIZE_PIXEL

    colored_img = np.zeros((physical_rgb.shape[0], physical_rgb.shape[1], 3), dtype=np.uint8)

    for ch in range(3):
        channel_data = physical_rgb[:, :, ch].astype(np.float32) / 255.0
        cmap = cm.get_cmap(colormap_names[ch])
        mapped = cmap(channel_data)[:, :, :3]
        colored_img += (mapped * 255).astype(np.uint8)

    colored_img = np.clip(colored_img, 0, 255).astype(np.uint8)

    img = Image.fromarray(colored_img, mode='RGB')
    if not KEEP_ASPECT_RATIO:
        img = img.resize((target_w, target_h), RESIZE_MODE)
    else:
        time_len = img.width
        freq_len = img.height
        scale = target_h / freq_len
        new_time_len = int(time_len * scale)
        img = img.resize((new_time_len, target_h), RESIZE_MODE)
        if new_time_len < target_w:
            pad_total = target_w - new_time_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img = ImageOps.expand(img, (pad_left, 0, pad_right, 0), fill=0)
        else:
            crop_left = (new_time_len - target_w) // 2
            img = img.crop((crop_left, 0, crop_left + target_w, target_h))

    img.save(save_path)


def save_colored_single_channel(channel_2d, save_path, cmap_name='Reds'):
    channel_data = channel_2d.astype(np.float32) / 255.0
    channel_data = np.transpose(channel_data, (1, 0))

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(channel_data)[:, :, :3]
    colored_uint8 = (colored * 255).astype(np.uint8)

    target_w, target_h = IMAGE_SIZE_PIXEL
    img = Image.fromarray(colored_uint8, mode='RGB')
    if not KEEP_ASPECT_RATIO:
        img = img.resize((target_w, target_h), RESIZE_MODE)
    else:
        time_len = img.width
        freq_len = img.height
        scale = target_h / freq_len
        new_time_len = int(time_len * scale)
        img = img.resize((new_time_len, target_h), RESIZE_MODE)
        if new_time_len < target_w:
            pad_total = target_w - new_time_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            img = ImageOps.expand(img, (pad_left, 0, pad_right, 0), fill=0)
        else:
            crop_left = (new_time_len - target_w) // 2
            img = img.crop((crop_left, 0, crop_left + target_w, target_h))
    img.save(save_path)


def single_audio_debug():
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TEST_AUDIO_PATH):
        print(f"❌ 错误：测试音频不存在，请修改 TEST_AUDIO_PATH 变量：{TEST_AUDIO_PATH}")
        return

    audio_filename = os.path.basename(TEST_AUDIO_PATH)
    audio_basename = os.path.splitext(audio_filename)[0]

    print("=" * 70)
    print(f"📌 单音频调试模式 (DeepShip 数据集)")
    print(f"🎨 三通道模式：{'开启' if USE_RGB_THREE_CHANNEL else '关闭'} | 模式：{RGB_CHANNEL_MODE if USE_RGB_THREE_CHANNEL else '灰度'}")
    print(f"📈 优化项：Mel频率轴{'开启' if USE_MEL_FREQ_AXIS else '关闭'} | 谱线增强{'开启' if USE_SPECTRAL_ENHANCE else '关闭'}")
    print(f"🧠 聚合策略：{'全局能量择优(新)' if USE_GLOBAL_TOPK else '局部像素择优(旧)'}")
    print(f"🖼️  图像尺寸：{IMAGE_SIZE_PIXEL[0]}×{IMAGE_SIZE_PIXEL[1]}")
    print(f"🌈 测试可视化：{'伪彩色映射 (类似Mel)' if TEST_USE_COLORMAP else '物理灰度图'}")
    print("=" * 70)

    try:
        y, _ = librosa.load(TEST_AUDIO_PATH, sr=TARGET_SR, mono=True)
        feature_gen = ChirpletFeatureGenerator(sr=TARGET_SR)

        window_samples = int(WINDOW_DURATION * TARGET_SR)
        step_samples = window_samples - int(OVERLAP_DURATION * TARGET_SR)

        start = 0
        count = 0
        max_test_segments = 3

        while start + window_samples <= len(y) and count < max_test_segments:
            seg = y[start:start + window_samples]
            feat = feature_gen.compute_feature(seg)

            save_path = os.path.join(TEST_OUTPUT_DIR, f"{audio_basename}_seg{count + 1:03d}.png")
            if TEST_USE_COLORMAP and USE_RGB_THREE_CHANNEL and RGB_CHANNEL_MODE == 'chirplet_band':
                apply_colormap_to_physical_rgb(feat, save_path, colormap_names=('Reds', 'Greens', 'Blues'))
                print(f"  ✅ 生成伪彩色主图：{os.path.basename(save_path)}")
            else:
                save_pure_feature(feat, save_path)
                print(f"  ✅ 生成物理主图：{os.path.basename(save_path)}")

            if SAVE_CHANNEL_SPLITS and USE_RGB_THREE_CHANNEL and RGB_CHANNEL_MODE == 'chirplet_band':
                channel_names = ['low_chirp', 'mid_chirp', 'high_chirp']
                channel_cmaps = ['Reds', 'Greens', 'Blues']
                for channel_idx, (channel_name, cmap_name) in enumerate(zip(channel_names, channel_cmaps)):
                    channel_feat = feat[:, :, channel_idx]
                    channel_save_path = os.path.join(
                        TEST_OUTPUT_DIR,
                        f"{audio_basename}_seg{count + 1:03d}_{channel_name}.png"
                    )
                    if TEST_USE_COLORMAP:
                        save_colored_single_channel(channel_feat, channel_save_path, cmap_name)
                        print(f"     ✅ 生成彩色通道图（{cmap_name}）：{os.path.basename(channel_save_path)}")
                    else:
                        save_pure_feature(channel_feat, channel_save_path)
                        print(f"     ✅ 生成灰度通道图：{os.path.basename(channel_save_path)}")

            start += step_samples
            count += 1

        print("\n" + "=" * 70)
        print(f"✅ 调试完成！共生成 {count} 个片段")
        if SAVE_CHANNEL_SPLITS and USE_RGB_THREE_CHANNEL and RGB_CHANNEL_MODE == 'chirplet_band':
            print(f"   包含 {count} 张主图 + {count * 3} 张三通道拆分图")
        print(f"保存到：{os.path.abspath(TEST_OUTPUT_DIR)}")
        print("=" * 70)

    except Exception as e:
        print(f"❌ 调试失败：{str(e)}")
        import traceback
        traceback.print_exc()


class AudioProcessor:
    def __init__(self, target_sr=16000, window_duration=0.5, overlap_duration=0.3):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.chirplet_generator = None

    def process_audio(self, audio_path, output_dir, audio_basename, max_segments=None):
        try:
            y, sr_actual = librosa.load(audio_path, sr=self.target_sr, mono=True)
            y = y.astype(np.float32)

            if self.chirplet_generator is None or self.chirplet_generator.sr != sr_actual:
                self.chirplet_generator = ChirpletFeatureGenerator(sr=sr_actual)

            window_samples = int(self.window_duration * sr_actual)
            overlap_samples = int(self.overlap_duration * sr_actual)
            step_samples = window_samples - overlap_samples

            if step_samples <= 0:
                step_samples = window_samples // 2

            segment_idx = 0
            total_samples = len(y)
            start_sample = 0
            expected_segments = max(1, (total_samples - window_samples) // step_samples + 1)
            processed_segments = 0

            while start_sample + window_samples <= total_samples:
                if max_segments is not None and processed_segments >= max_segments:
                    break

                y_segment = y[start_sample:start_sample + window_samples]
                feature_norm = self.chirplet_generator.compute_feature(y_segment)

                save_filename = f"{audio_basename}_chirplet_seg{segment_idx + 1:03d}.png"
                save_path = os.path.join(output_dir, save_filename)
                save_pure_feature(feature_norm, save_path)

                processed_segments += 1
                if segment_idx % 5 == 0 and segment_idx > 0:
                    print(f"    Generated {segment_idx + 1}/{expected_segments} segments")

                segment_idx += 1
                start_sample += step_samples

            print(f"  ✓ Successfully generated {processed_segments} chirplet feature segments")
            return processed_segments

        except Exception as e:
            print(f"  ✗ Failed to process audio: {e}")
            import traceback
            traceback.print_exc()
            return 0


# ==============================================================================

# ==============================================================================
def batch_process_chirplet_features():
    import shutil
    import random

    if RUN_MODE == 'test':
        max_files_per_class = 1
        print(f"🔍 运行模式：TEST（每类只处理 1 个文件）")
    else:
        max_files_per_class = MAX_FILES_PER_CLASS
        print(f"🚀 运行模式：FULL（每类最多处理 {MAX_FILES_PER_CLASS} 个文件）")

    if os.path.exists(OUTPUT_ROOT_DIR):
        shutil.rmtree(OUTPUT_ROOT_DIR)
        print(f"Cleared existing directory: {OUTPUT_ROOT_DIR}")
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    unique_classes = list(set(CLASS_MAPPING.values()))
    for cls in unique_classes:
        cls_dir = os.path.join(OUTPUT_ROOT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
    print(f"Created {len(unique_classes)} class folders")


    audio_files = []
    for class_folder in os.listdir(AUDIO_ROOT_DIR):
        class_path = os.path.join(AUDIO_ROOT_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        if class_folder not in CLASS_MAPPING:
            print(f"⚠️  跳过未知类别文件夹: {class_folder}")
            continue
        
        mapped_class = CLASS_MAPPING[class_folder]
        
        for filename in os.listdir(class_path):
            if filename.lower().endswith('.wav'):
                audio_path = os.path.join(class_path, filename)
                audio_files.append({
                    'path': audio_path,
                    'filename': filename,
                    'original_class': class_folder,
                    'mapped_class': mapped_class
                })

    print(f"Successfully scanned DeepShip dataset, total {len(audio_files)} audio records")

    audio_processor = AudioProcessor(
        target_sr=TARGET_SR,
        window_duration=WINDOW_DURATION,
        overlap_duration=OVERLAP_DURATION
    )

    class_stats = {cls: {'files': 0, 'segments': 0} for cls in unique_classes}
    total_processed = 0
    total_segments = 0

    print("\n" + "=" * 70)
    print(f"Chirplet Feature Batch Processing System (DeepShip - Sampled)")
    print("=" * 70)
    print(f"【抽样设置】")
    print(f"  每类最大音频数：{MAX_FILES_PER_CLASS}")
    print(f"  随机种子：{SAMPLING_RANDOM_SEED}")
    print(f"\n【对齐原论文的固定参数】")
    print(f"  Sample Rate: {TARGET_SR} Hz")
    print(f"  Window Duration: {WINDOW_DURATION}s")
    print(f"  Overlap Duration: {OVERLAP_DURATION}s")
    print(f"  Image Size: {IMAGE_SIZE_PIXEL[0]}×{IMAGE_SIZE_PIXEL[1]}")
    print(f"\n【优化后的创新参数】")
    print(f"  Chirplet Window: {CHIRP_WIN_LENGTH} | Fmax: {CHIRP_FMAX}")
    print(f"  Chirp Rate Range: {CHIRP_C_RANGE} | Top-K: {CHIRP_AGG_TOPK}")
    print(f"  Aggregation: {'Global Energy-based' if USE_GLOBAL_TOPK else 'Local Pixel-based'}")
    print(f"  RGB Mode: {'Enabled (' + RGB_CHANNEL_MODE + ')' if USE_RGB_THREE_CHANNEL else 'Disabled'}")
    print(f"  Gamma Correction: {ENHANCE_GAMMA}")
    print("=" * 70)

    # Process files by class.
    for target_class in tqdm(unique_classes, desc="Processing Progress"):
        print(f"\nProcessing class: {target_class}")
        class_files = [f for f in audio_files if f['mapped_class'] == target_class]

        if len(class_files) == 0:
            print(f"  Warning: No audio found for class '{target_class}'")
            continue


        if len(class_files) > max_files_per_class:
            # Set the random seed
            random.seed(SAMPLING_RANDOM_SEED)
            class_files = random.sample(class_files, max_files_per_class)
            print(f"  抽样：从 {len(class_files) + (len(class_files) if len(class_files) < max_files_per_class else 0)} 个中随机选择 {max_files_per_class} 个")

        output_dir = os.path.join(OUTPUT_ROOT_DIR, target_class)

        for idx, audio_info in enumerate(class_files, 1):
            audio_path = audio_info['path']
            audio_filename = audio_info['filename']
            original_class = audio_info['original_class']

            if not os.path.exists(audio_path):
                print(f"  Audio file not found [{idx}/{len(class_files)}]: {audio_path}")
                continue

            print(f"  Processing [{idx}/{len(class_files)}]: {audio_filename}")
            audio_basename = os.path.splitext(audio_filename)[0]
            segments_generated = audio_processor.process_audio(
                audio_path, output_dir, audio_basename
            )

            if segments_generated > 0:
                total_processed += 1
                total_segments += segments_generated
                class_stats[target_class]['files'] += 1
                class_stats[target_class]['segments'] += segments_generated

    print("\n" + "=" * 70)
    print("Batch Processing Completed!")
    print(f"Successfully processed audio files: {total_processed}")
    print(f"Total images generated: {total_segments}")
    print(f"Output directory: {os.path.abspath(OUTPUT_ROOT_DIR)}")

    print("\nClass Statistics:")
    print("-" * 40)
    for cls in sorted(unique_classes):
        stats = class_stats[cls]
        if stats['files'] > 0:
            avg_segments = stats['segments'] / stats['files'] if stats['files'] > 0 else 0
            print(f"  {cls}:")
            print(f"    Audio files: {stats['files']}")
            print(f"    Feature images: {stats['segments']}")
            print(f"    Average per audio: {avg_segments:.1f}")



if __name__ == "__main__":
    print("=" * 70)
    print("Underwater Acoustic Signal Feature Extraction System - Chirplet Version (DeepShip Sampled)")
    print("=" * 70)

    required_libs = ['librosa', 'pandas', 'matplotlib', 'numpy', 'tqdm', 'scipy', 'PIL']
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)

    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install command: pip install " + " ".join(missing))
        exit(1)

    if RUN_MODE == 'test':
        print("\n" + "=" * 70)
        print("🔍 运行模式：TEST（单音频调试）")
        print("=" * 70)
        single_audio_debug()
        print("\n✅ 测试完成，程序退出。")

    elif RUN_MODE == 'full':
        print("\n" + "=" * 70)
        print("🚀 运行模式：FULL（每类抽样处理）")
        print("=" * 70)
        batch_process_chirplet_features()

    else:
        print(f"❌ 错误：RUN_MODE 只能是 'test' 或 'full'，当前是：{RUN_MODE}")
        exit(1)