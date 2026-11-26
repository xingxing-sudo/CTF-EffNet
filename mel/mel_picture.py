# Use a headless plotting backend
import matplotlib
matplotlib.use('Agg')

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


# Data paths
AUDIO_ROOT_DIR = "./data/ShipsEar"
CSV_FILE_PATH = "./data/ShipsEar/shipsEarMeta_90.csv"
OUTPUT_ROOT_DIR = "./mel/mel_picture/all_mel_picture"


TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.3


N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 8000


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


class MelSpectrogramGenerator:
    
    def __init__(self, sr=16000, n_fft=2048, hop_length=501, 
                 n_mels=128, fmin=50, fmax=8000):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = min(fmax, sr // 2)
        

        self.mel_filter = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
    def compute_mel_spectrogram(self, audio):

        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann'
        )
        
        # Magnitude spectrum
        spectrogram = np.abs(stft)
        

        mel_spec = np.dot(self.mel_filter, spectrogram)
        
        # Convert magnitudes to decibels.
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db


def save_colored_mel_image(mel_spec_db, save_path):

    fig_size = (IMAGE_SIZE_PIXEL[0] / DPI, IMAGE_SIZE_PIXEL[1] / DPI)
    

    fig = plt.figure(figsize=fig_size, dpi=DPI)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    

    librosa.display.specshow(
        mel_spec_db,
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis=None,
        y_axis=None,
        fmin=FMIN,
        fmax=FMAX,
        ax=ax,
        cmap=COLOR_MAP
    )
    

    plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
    
    return True

# Audio processing
class AudioProcessor:
    
    def __init__(self, target_sr=TARGET_SR, window_duration=WINDOW_DURATION, overlap_duration=OVERLAP_DURATION):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.mel_generator = None
        
    def process_audio(self, audio_path, output_dir, audio_basename, 
                      max_segments=None):
        try:
            # Load audio.
            y, sr_actual = librosa.load(audio_path, sr=self.target_sr, mono=True)
            y = y.astype(np.float32)
            

            if self.mel_generator is None or self.mel_generator.sr != sr_actual:
                self.mel_generator = MelSpectrogramGenerator(
                    sr=sr_actual,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    n_mels=N_MELS,
                    fmin=FMIN,
                    fmax=FMAX
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
            
            # Process all audio segments.
            processed_segments = 0
            
            while start_sample + window_samples <= total_samples:

                if max_segments is not None and processed_segments >= max_segments:
                    break
                
                # Extract segments.
                y_segment = y[start_sample:start_sample + window_samples]
                

                mel_spec_db = self.mel_generator.compute_mel_spectrogram(y_segment)
                

                save_filename = f"{audio_basename}_mel_seg{segment_idx + 1:03d}.png"
                save_path = os.path.join(output_dir, save_filename)
                
                success = save_colored_mel_image(mel_spec_db, save_path)
                
                if success:
                    processed_segments += 1
                    

                    if segment_idx % 5 == 0 and segment_idx > 0:
                        print(f"    Generated {segment_idx + 1}/{expected_segments} segments")
                
                segment_idx += 1
                start_sample += step_samples
            
            print(f"  ✓ Successfully generated {processed_segments} colored mel spectrograms (224x224, {COLOR_MAP})")
            return processed_segments
            
        except Exception as e:
            print(f"  ✗ Failed to process audio: {e}")
            import traceback
            traceback.print_exc()
            return 0

# Batch processing
def batch_process_mel_spectrograms(max_files_per_class=None, max_segments_per_audio=None):
    import shutil
    
    # Create output directories
    output_dir_name = OUTPUT_ROOT_DIR
    
    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
        print(f"Cleared existing directory: {output_dir_name}")
    os.makedirs(output_dir_name, exist_ok=True)
    
    # Create class directories.
    unique_classes = list(set(CLASS_MAPPING.values()))
    for cls in unique_classes:
        cls_dir = os.path.join(output_dir_name, cls)
        os.makedirs(cls_dir, exist_ok=True)
    print(f"Created {len(unique_classes)} class folders")
    
    # Read the metadata CSV.
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded CSV file, total {len(df)} audio records")
        
        if 'filename' not in df.columns or 'type' not in df.columns:
            raise ValueError("CSV file must contain 'filename' and 'type' columns")
            
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return
    
    # Initialize the processor.
    audio_processor = AudioProcessor(
        target_sr=TARGET_SR,
        window_duration=WINDOW_DURATION,
        overlap_duration=OVERLAP_DURATION
    )
    
    # Summary statistics
    class_stats = {cls: {'files': 0, 'segments': 0} for cls in unique_classes}
    total_processed = 0
    total_segments = 0
    
    print("\n" + "=" * 70)
    print(f"Colored Mel Spectrogram Batch Processing System (224x224, {COLOR_MAP} Colormap, No Text)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Sample Rate: {TARGET_SR} Hz")
    print(f"  Window Duration: {WINDOW_DURATION}s")
    print(f"  Overlap Duration: {OVERLAP_DURATION}s")
    print(f"  Mel N_FFT: {N_FFT}")
    print(f"  Mel Hop Length: {HOP_LENGTH} (论文明确值，已修复！)")
    print(f"  Output Image Size: {IMAGE_SIZE_PIXEL[0]}x{IMAGE_SIZE_PIXEL[1]}")
    print(f"  Colormap: {COLOR_MAP} (和你原来代码一样的彩色！)")
    print("=" * 70)
    
    # Process files by class.
    df['mapped_class'] = df['type'].map(CLASS_MAPPING)
    
    for target_class in tqdm(unique_classes, desc="Processing Progress"):
        print(f"\nProcessing class: {target_class}")
        

        class_df = df[df['mapped_class'] == target_class].copy()
        
        if len(class_df) == 0:
            print(f"  Warning: No audio found for class '{target_class}'")
            continue
            
        if max_files_per_class is not None:
            class_df = class_df.head(max_files_per_class)
            print(f"  Limiting to first {max_files_per_class} files")
        
        output_dir = os.path.join(output_dir_name, target_class)
        

        for idx, row in enumerate(class_df.itertuples(), 1):
            audio_filename = row.filename
            original_class = row.type
            

            audio_path = os.path.join(AUDIO_ROOT_DIR, audio_filename)
            
            if not os.path.exists(audio_path):
                print(f"  Audio file not found [{idx}/{len(class_df)}]: {audio_path}")
                continue
            

            print(f"  Processing [{idx}/{len(class_df)}]: {audio_filename}")
            
            audio_basename = os.path.splitext(audio_filename)[0]
            segments_generated = audio_processor.process_audio(
                audio_path, output_dir, audio_basename, 
                max_segments_per_audio
            )
            
            if segments_generated > 0:
                total_processed += 1
                total_segments += segments_generated
                class_stats[target_class]['files'] += 1
                class_stats[target_class]['segments'] += segments_generated
    
    # Final counts
    print("\n" + "=" * 70)
    print("Batch Processing Completed!")
    print(f"Successfully processed audio files: {total_processed}/{len(df)}")
    print(f"Total colored 224x224 images generated: {total_segments}")
    print(f"Output directory: {os.path.abspath(output_dir_name)}")
    

    print("\nClass Statistics:")
    print("-" * 40)
    for cls in sorted(unique_classes):
        stats = class_stats[cls]
        if stats['files'] > 0:
            avg_segments = stats['segments'] / stats['files'] if stats['files'] > 0 else 0
            print(f"  {cls}:")
            print(f"    Audio files: {stats['files']}")
            print(f"    Colored 224x224 images: {stats['segments']}")
            print(f"    Average per audio: {avg_segments:.1f}")

# Main function
if __name__ == "__main__":
    print("=" * 70)
    print("Underwater Acoustic Signal Feature Extraction System - Colored Mel 224x224 Version")
    print("=" * 70)
    
    # Check dependencies.
    required_libs = ['librosa', 'pandas', 'matplotlib', 'numpy', 'tqdm', 'PIL']
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
    
    # Start batch processing.
    print(f"\nStarting batch processing...")
    
    batch_process_mel_spectrograms(
        max_files_per_class=None,  # Process all files.
        max_segments_per_audio=None  # Process all audio segments.
    )