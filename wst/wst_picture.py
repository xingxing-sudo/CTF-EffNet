# Use a headless plotting backend
import matplotlib
matplotlib.use('Agg')

import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


AUDIO_ROOT_DIR = "./data/ShipsEar"
CSV_FILE_PATH = "./data/ShipsEar/shipsEarMeta_90.csv"
OUTPUT_ROOT_DIR = "./wst/wst_picture/all_wst_picture"


TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.3


WST_J = 6
WST_Q = 8
WST_ORDER = 2


IMAGE_SIZE_PIXEL = (224, 224)
COLOR_MAP = 'magma'
DPI = 100

# Class mapping
CLASS_MAPPING = {
    "Ocean liner": "A_Large_Merchant_Ship",
    "RORO": "A_Large_Merchant_Ship",
    "Dredger": "A_Large_Merchant_Ship",
    "Motorboat": "B_Small_Speedboat",
    "Tugboat": "B_Small_Speedboat",
    "Pilot ship": "B_Small_Speedboat",
    "Sailboat": "B_Small_Speedboat",
    "Yacht": "B_Small_Speedboat",
    "High speed motorboat": "B_Small_Speedboat",
    "Zodiac": "B_Small_Speedboat",
    "Fishboat": "C_Fishing_Boat",
    "Trawler": "C_Fishing_Boat",
    "Mussel boat": "C_Fishing_Boat",
    "fishboat": "C_Fishing_Boat",
    "Passengers": "D_Ferry",
    "Natural ambient noise": "E_Ambient_Noise"
}


class WstSpectrogramGenerator:

    def __init__(self, sr=16000, signal_length=8000, J=6, Q=8, order=2):
        self.sr = sr
        self.signal_length = signal_length
        self.J = J
        self.Q = Q
        self.order = order

        from kymatio.numpy import Scattering1D
        self.scattering = Scattering1D(
            J=J,
            shape=(signal_length,),
            Q=Q,
            max_order=order,
            average=True
        )


        self.meta = self.scattering.meta()
        self.order0_idx = np.where(self.meta['order'] == 0)[0]
        self.order1_idx = np.where(self.meta['order'] == 1)[0]
        self.order2_idx = np.where(self.meta['order'] == 2)[0]

        print(f"✅ WST初始化完成：")
        print(f"   零阶：{len(self.order0_idx)}个，一阶：{len(self.order1_idx)}个，二阶：{len(self.order2_idx)}个")

    def compute_wst_spectrogram(self, audio, mode='train'):

        if len(audio) > self.signal_length:
            audio = audio[:self.signal_length]
        else:
            audio = np.pad(audio, (0, self.signal_length - len(audio)), mode='constant')


        Sx = np.abs(self.scattering(audio))


        S0 = Sx[self.order0_idx]
        S1 = Sx[self.order1_idx]
        S2 = Sx[self.order2_idx]


        if mode == 'paper_second':

            coeffs_db = librosa.amplitude_to_db(S2, ref=np.max, top_db=70.0)
            mat = coeffs_db
        elif mode == 'paper_first':

            coeffs_db = librosa.amplitude_to_db(S1, ref=np.max, top_db=70.0)
            mat = coeffs_db
        else:

            S_combined = np.vstack([S0, S1, S2])
            coeffs_db = librosa.amplitude_to_db(S_combined, ref=np.max, top_db=70.0)
            mat = coeffs_db


        p2, p98 = np.percentile(mat, (2, 98))
        mat_clipped = np.clip(mat, p2, p98)
        mat_norm = (mat_clipped - p2) / (p98 - p2 + 1e-8)


        mat_norm = np.flipud(mat_norm)

        return mat_norm


def save_colored_wst_image(wst_mat, save_path):
    fig_size = (IMAGE_SIZE_PIXEL[0] / DPI, IMAGE_SIZE_PIXEL[1] / DPI)
    fig = plt.figure(figsize=fig_size, dpi=DPI)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    ax.imshow(
        wst_mat,
        cmap=COLOR_MAP,
        aspect='auto',
        origin='lower',
        interpolation='bicubic',
        vmin=0, vmax=1
    )

    plt.savefig(
        save_path,
        dpi=DPI,
        bbox_inches='tight',
        pad_inches=0,
        facecolor='black'
    )
    plt.close(fig)
    return True

# Audio processing
class AudioProcessor:
    def __init__(self, target_sr=TARGET_SR, window_duration=WINDOW_DURATION, overlap_duration=OVERLAP_DURATION):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.wst_generator = None

    def process_audio(self, audio_path, output_dir, audio_basename, mode='train', max_segments=None):
        try:
            y, sr_actual = librosa.load(audio_path, sr=self.target_sr, mono=True)
            y = y.astype(np.float32)

            if self.wst_generator is None or self.wst_generator.sr != sr_actual:
                self.wst_generator = WstSpectrogramGenerator(
                    sr=sr_actual,
                    signal_length=int(WINDOW_DURATION * sr_actual),
                    J=WST_J,
                    Q=WST_Q,
                    order=WST_ORDER
                )

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
                
                wst_spec = self.wst_generator.compute_wst_spectrogram(y_segment, mode=mode)

                save_filename = f"{audio_basename}_wst_seg{segment_idx + 1:03d}.png"
                save_path = os.path.join(output_dir, save_filename)

                success = save_colored_wst_image(wst_spec, save_path)

                if success:
                    processed_segments += 1
                    if segment_idx % 10 == 0 and segment_idx > 0:
                        print(f"    Generated {segment_idx + 1}/{expected_segments} segments")

                segment_idx += 1
                start_sample += step_samples

            print(f"  ✓ Successfully generated {processed_segments} WST spectrograms")
            return processed_segments

        except Exception as e:
            print(f"  ✗ Failed to process audio: {e}")
            import traceback
            traceback.print_exc()
            return 0

# Batch processing
def batch_process_wst_spectrograms(mode='train', max_files_per_class=None, max_segments_per_audio=None):
    import shutil

    output_dir_name = OUTPUT_ROOT_DIR

    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
        print(f"Cleared existing directory: {output_dir_name}")
    os.makedirs(output_dir_name, exist_ok=True)

    unique_classes = list(set(CLASS_MAPPING.values()))
    for cls in unique_classes:
        cls_dir = os.path.join(output_dir_name, cls)
        os.makedirs(cls_dir, exist_ok=True)
    print(f"Created {len(unique_classes)} class folders")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded CSV file, total {len(df)} audio records")
        if 'filename' not in df.columns or 'type' not in df.columns:
            raise ValueError("CSV file must contain 'filename' and 'type' columns")
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return

    audio_processor = AudioProcessor()

    class_stats = {cls: {'files': 0, 'segments': 0} for cls in unique_classes}
    total_processed = 0
    total_segments = 0

    print("\n" + "=" * 70)
    print(f"🔥 论文级WST频谱图生成系统 (Liu et al., 2025)")
    print("=" * 70)
    print(f"运行模式：{mode}")
    print(f"配置：J={WST_J}, Q={WST_Q}, order={WST_ORDER}")
    print(f"输出尺寸：{IMAGE_SIZE_PIXEL[0]}×{IMAGE_SIZE_PIXEL[1]}")
    print("=" * 70)

    df['mapped_class'] = df['type'].map(CLASS_MAPPING)

    for target_class in tqdm(unique_classes, desc="处理进度"):
        print(f"\n处理类别：{target_class}")

        class_df = df[df['mapped_class'] == target_class].copy()

        if len(class_df) == 0:
            print(f"  警告：未找到类别 '{target_class}' 的音频文件")
            continue

        if max_files_per_class is not None:
            class_df = class_df.head(max_files_per_class)
            print(f"  限制处理前 {max_files_per_class} 个文件")

        output_dir = os.path.join(output_dir_name, target_class)

        for idx, row in enumerate(class_df.itertuples(), 1):
            audio_filename = row.filename
            audio_path = os.path.join(AUDIO_ROOT_DIR, audio_filename)

            if not os.path.exists(audio_path):
                print(f"  文件不存在 [{idx}/{len(class_df)}]: {audio_path}")
                continue

            print(f"  处理 [{idx}/{len(class_df)}]: {audio_filename}")

            audio_basename = os.path.splitext(audio_filename)[0]
            segments_generated = audio_processor.process_audio(
                audio_path, output_dir, audio_basename,
                mode=mode,
                max_segments=max_segments_per_audio
            )

            if segments_generated > 0:
                total_processed += 1
                total_segments += segments_generated
                class_stats[target_class]['files'] += 1
                class_stats[target_class]['segments'] += segments_generated

    print("\n" + "=" * 70)
    print("批量处理完成！")
    print(f"成功处理：{total_processed}个文件，生成{total_segments}张图")
    print(f"输出目录：{os.path.abspath(output_dir_name)}")

# Main function
if __name__ == "__main__":
    print("=" * 70)
    print("水声信号特征提取系统 - 论文级WST版本（丝滑版）")
    print("=" * 70)

    # Check dependencies.
    required_libs = ['librosa', 'pandas', 'matplotlib', 'numpy', 'tqdm', 'kymatio']
    missing = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"缺少依赖：{', '.join(missing)}")
        print(f"安装：pip install {' '.join(missing)}")
        exit(1)





    print("\n开始批量处理（训练模式）...")
    batch_process_wst_spectrograms(mode='train')