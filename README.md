# CTF-EffNet

CTF-EffNet is an experimental implementation for underwater vessel classification. The main ShipsEar pipeline produces three-channel chirplet features and trains an EfficientNetV2-B0 classifier with a chirp time-frequency fusion (CTF) module. The repository also contains ResNet-18 experiments, CQT/Mel/STFT/WST baselines, model and chirp-band ablations, noise tests, and a mini DeepShip experiment.

[Reproducibility review](REPRODUCIBILITY.md) · [Intended GitHub repository](https://github.com/xingxing-sudo/CTF-EffNet)

## Environment

Run the commands below from the repository root on Linux with Python 3.12. Use a CUDA-capable NVIDIA system for full training. The requirements pin the direct dependencies observed in the preparation environment. The historical training environment was not captured as a complete lockfile.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
~~~

For CPU evaluation, use the matching wheels from the PyTorch CPU index at https://download.pytorch.org/whl/cpu. GPU-memory logging is optional and uses the pynvml module. Audio decoding uses SoundFile/libsndfile through librosa.

## Data and features

- Request the full **ShipsEar** collection from the dataset maintainers. The official site provides sample downloads and the full-access contact. Put the WAV files in **data/ShipsEar/** with the names listed in the bundled **shipsEarMeta_90.csv**. The extraction scripts use this metadata for filename and category mapping. [ShipsEar dataset](https://underwaternoise.atlanttic.uvigo.es/)
- For **DeepShip**, follow the repository's access instructions. Put the original class folders under **deepship/data/Deepship/** and check CLASS_MAPPING in the generator. The historical mini protocol chooses up to 15 recordings per class. [DeepShip dataset](https://github.com/irfankamboh/DeepShip)

The repository supplies file membership lists and ShipsEar metadata. Obtain audio separately under the dataset providers' terms. Generated feature images and model weights are stored locally outside the release. See the review for metadata redistribution and exact historical DeepShip membership.

Generate the main feature representation and restore the on-disk historical split:

~~~bash
python chirplet/chirplet_picture_pro.py
python tools/materialize_split.py chirplet --link
~~~

Use a clean split destination. Omit the link flag to copy feature images on filesystems that lack symlink support. The script reports missing filenames before writing. The resulting small-sample directory contains **train/**, **val/** and **test/** under **chirplet/chirplet_picture_pro/chirplet_shipsear_small_10_train/**. The final holdout lives under **chirplet/chirplet_picture_pro/chirplet_shipsear_small_90_test/**. The original generation/selection scripts and membership lists remain in each **picture_split/** directory for reference. Use the released manifest to recover the recorded partition rather than re-running the shuffle.

## Training and testing

~~~bash
python 'chirplet/efficientnetv2_b0+CTF.py'
# Set this to the timestamped best_model.pth produced by training.
CTF_CHECKPOINT=/absolute/path/to/best_model.pth python 'chirplet/efficientnetv2_b0+CTF_test.py'
~~~

Training writes a timestamped run under **chirplet/logs/tf_efficientnetv2_b0+CTF/small_10/** and uses the validation set for checkpoint selection. The standalone evaluator defaults to the separate final holdout. It writes metrics and a confusion matrix next to the checkpoint. To evaluate an alternative ImageFolder tree, set CTF_TEST_DATA to its root. CTF_WORKERS=0 uses single-process data loading for the main CTF evaluator. Use CTF_BATCH_SIZE=1 for a low-memory CPU test. Defaults are eight workers and batch size 128. Load only checkpoints you obtained from a trusted source.

The main training script records seed 456, 300 epochs, batch size 64, 224×224 inputs, AdamW at an initial learning rate of 0.0001, ten warm-up epochs, cosine scheduling, and MixUp/CutMix. The feature generator records 16 kHz audio, a 0.5 s window with 0.3 s overlap, and chirp-rate index bands [0, 21), [21, 43), [43, 64). Each script's settings are authoritative for that experiment.

| Experiment | Entrypoints and inputs |
| --- | --- |
| ShipsEar feature baselines | Generate using **cqt/cqt_Picture.py**, **mel/mel_picture.py**, **stft/stft_Picture.py** or **wst/wst_picture.py**. Run **python tools/materialize_split.py METHOD --link** with the corresponding lowercase method. Run that method's **efficientnetv2_b0.py** or **resnet18.py** and evaluate using **test.py** or **resnet18_test.py** with CTF_CHECKPOINT. |
| Chirplet backbone baselines | Use the same Chirplet split with **chirplet/efficientnetv2_b0.py**, **chirplet/resnet18.py**, or **chirplet/resnet18+CTF.py**. The evaluation scripts are **chirplet/test.py**, **chirplet/resnet18_test.py**, and **chirplet/resnet18+CTF_test.py**, respectively. |
| CTF component ablation | Run the three **train_ctf_*.py** scripts under **chirplet/logs/ctf_ablation/**. Set the three checkpoint paths in **test_all_ablation.py** in that directory to the checkpoints of the new runs before evaluation. |
| Chirp-band ablation | Generate features with **chirplet/band_partition/chirplet_feature_generator.py**. Seed the partition using the commands below. This run trains and evaluates all five schemes. |
| Gaussian noise evaluation | Run **chirplet/efficientnetv2_b0+CTF+SNR_test.py** with CTF_CHECKPOINT. The archived noise protocol adds Gaussian noise to image tensors. Read the review before interpreting its relation to audio-domain SNR or recording-independent tests. |
| Full-data ShipsEar | Run **full_data/chirplet/efficientnetv2_b0+CTF.py** after generating Chirplet features. Training and evaluation use the script's in-memory split. |
| Mini DeepShip | Run **deepship/chirplet/chirplet_picture.py** followed by **deepship/chirplet/efficientnetv2_b0+CTF.py**. The latter trains and evaluates its in-memory split. |
| Analysis and plots | See **z_comparison/** for figure and table scripts and **reference_results/** for historical numeric results and training records. Feature and embedding plots require their inputs or checkpoints. |

~~~bash
python chirplet/band_partition/chirplet_feature_generator.py
mkdir -p chirplet/band_partition/ablation_splits/band_1bin
cp splits/band_ablation_split_info.json chirplet/band_partition/ablation_splits/band_1bin/split_info.json
python chirplet/band_partition/run_ablation.py
~~~

The band runner uses symbolic links. Its training and evaluation outputs are generated under **chirplet/band_partition/ablation_logs/**.

## Availability statements for the manuscript

**Data availability.** ShipsEar is available from its maintainers as described on the [ShipsEar dataset website](https://underwaternoise.atlanttic.uvigo.es/). DeepShip access is described in the [DeepShip repository](https://github.com/irfankamboh/DeepShip). The release contains the ShipsEar image membership of the archived experiments and the available reference results. Cite the datasets in the manuscript and describe any access conditions.

**Code availability.** Once the repository is approved and public, cite the version used for the manuscript at [CTF-EffNet](https://github.com/xingxing-sudo/CTF-EffNet). Supply a versioned repository permalink or archival DOI when available.

The statements above are drafts for the authors to verify against the final manuscript and [PAA submission guidelines](https://link.springer.com/journal/10044/submission-guidelines). Confirm the project license and metadata redistribution terms before publishing. See the review for open reproducibility and manuscript checks.
