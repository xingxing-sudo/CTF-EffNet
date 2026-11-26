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
warnings.filterwarnings('ignore')


# Data paths
AUDIO_ROOT_DIR = "./data/ShipsEar"
CSV_FILE_PATH = "./data/ShipsEar/shipsEarMeta_90.csv"
OUTPUT_ROOT_DIR = "./cqt/cqt_picture/all_cqt_picture"


TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.3


CQT_FMIN = 50
CQT_BINS_PER_OCTAVE = 12
CQT_N_BINS = 84
CQT_HOP_LENGTH = 512
CQT_WINDOW = 'hann'  # Window function


IMG_SIZE_PIXELS = 224
DPI = 100
COLOR_MAP = 'magma'

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


class CQTSpectrogramGenerator:
    
    def __init__(self, sr=16000, fmin=50, n_bins=84, bins_per_octave=12, 
                 hop_length=501, window='hann'):
        self.sr = sr
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.window = window
        
    def compute_cqt_spectrogram(self, audio):

        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            hop_length=self.hop_length,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            window=self.window
        )
        
        # Magnitude spectrum
        spectrogram = np.abs(cqt)
        
        # Convert magnitudes to decibels.
        cqt_spec_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        
        return cqt_spec_db

# Visualization
def plot_cqt_spectrogram_clean(cqt_spec_db, sr, hop_length, save_path=None):

    figsize = (IMG_SIZE_PIXELS / DPI, IMG_SIZE_PIXELS / DPI)
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    

    librosa.display.specshow(
        cqt_spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=None,  # Hide time-axis labels.
        y_axis=None,  # Hide frequency-axis labels.
        ax=ax,
        cmap=COLOR_MAP
    )
    
    # Hide axes, ticks and borders.
    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    if save_path:

        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    else:
        plt.show()
        return None

# Audio processing
class AudioProcessor:
    
    def __init__(self, target_sr=16000, window_duration=0.5, overlap_duration=0.3):
        self.target_sr = target_sr
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration
        self.cqt_generator = None
        
    def process_audio(self, audio_path, output_dir, audio_basename, 
                      max_segments=None):
        try:
            # Load audio.
            y, sr_actual = librosa.load(audio_path, sr=self.target_sr, mono=True)
            y = y.astype(np.float32)
            

            if self.cqt_generator is None or self.cqt_generator.sr != sr_actual:
                self.cqt_generator = CQTSpectrogramGenerator(
                    sr=sr_actual,
                    fmin=CQT_FMIN,
                    n_bins=CQT_N_BINS,
                    bins_per_octave=CQT_BINS_PER_OCTAVE,
                    hop_length=CQT_HOP_LENGTH,
                    window=CQT_WINDOW
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
                
                # Extract segments.
                y_segment = y[start_sample:start_sample + window_samples]
                

                save_filename = f"{audio_basename}_cqt_seg{segment_idx + 1:03d}.png"
                save_path = os.path.join(output_dir, save_filename)
                
                cqt_spec_db = self.cqt_generator.compute_cqt_spectrogram(y_segment)
                

                success = plot_cqt_spectrogram_clean(
                    cqt_spec_db, sr_actual, CQT_HOP_LENGTH,
                    save_path
                )
                
                if success:
                    processed_segments += 1
                    if segment_idx % 5 == 0 and segment_idx > 0:
                        print(f"    Generated {segment_idx + 1}/{expected_segments} segments")
                
                segment_idx += 1
                start_sample += step_samples
            
            print(f"  ✓ Successfully generated {processed_segments} CQT spectrogram segments (224x224 px, {COLOR_MAP})")
            return processed_segments
            
        except Exception as e:
            print(f"  ✗ Failed to process audio: {e}")
            import traceback
            traceback.print_exc()
            return 0

# Batch processing
def batch_process_cqt_spectrograms(max_files_per_class=None, max_segments_per_audio=None):
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
    print(f"CQT Spectrogram Batch Processing (224x224 pixels, {COLOR_MAP} colormap, clean image)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Sample Rate: {TARGET_SR} Hz")
    print(f"  Window Duration: {WINDOW_DURATION}s")
    print(f"  Overlap Duration: {OVERLAP_DURATION}s")
    print(f"  CQT Fmin: {CQT_FMIN} Hz")
    print(f"  CQT Bins/Octave: {CQT_BINS_PER_OCTAVE}")
    print(f"  CQT Total Bins: {CQT_N_BINS}")
    print(f"  CQT Hop Length: {CQT_HOP_LENGTH} (与原论文Mel一致！)")
    print(f"  Output Image Size: {IMG_SIZE_PIXELS}x{IMG_SIZE_PIXELS} pixels")
    print(f"  Colormap: {COLOR_MAP} (和你原来Mel代码一样的彩色！)")
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
    print(f"Total CQT images generated: {total_segments}")
    print(f"Output directory: {os.path.abspath(output_dir_name)}")
    
    print("\nClass Statistics:")
    print("-" * 40)
    for cls in sorted(unique_classes):
        stats = class_stats[cls]
        if stats['files'] > 0:
            avg_segments = stats['segments'] / stats['files'] if stats['files'] > 0 else 0
            print(f"  {cls}:")
            print(f"    Audio files: {stats['files']}")
            print(f"    CQT images: {stats['segments']}")
            print(f"    Average per audio: {avg_segments:.1f}")

# Main function
if __name__ == "__main__":
    print("=" * 70)
    print("Underwater Acoustic CQT Spectrogram Generator (224x224 Clean Colored Images)")
    print("=" * 70)
    
    # Check dependencies.
    required_libs = ['librosa', 'pandas', 'matplotlib', 'numpy', 'tqdm']
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
    print(f"\nStarting batch processing (CQT, 224x224 pixels, {COLOR_MAP})...")
    batch_process_cqt_spectrograms(
        max_files_per_class=None,  # Process all files.
        max_segments_per_audio=None  # Process all audio segments.
    )