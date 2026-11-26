# Reproducibility review

This review compares the release scripts with the local experiment artifacts and the available research notes. The manuscript was unavailable, so the final text, figures, tables and experimental claims still require an author cross-check. The intended repository could not be retrieved for comparison during preparation. This package is a local publication draft.

## Verified from the source artifacts

- For each of Chirplet, CQT, Mel, STFT and WST, the materialized small-sample directories contain 3,915 training images, 862 validation images and 869 internal test images. The separate holdout contains 50,843 images per method. Each released small-sample membership matches its original filename list. No relative image path occurs in multiple partitions for a given method.
- Segments from an individual recording therefore span the two sets. In Chirplet, 0.5 s windows overlap by 0.3 s. The historical scores support a segment-level protocol with shared recording sources. A claim about generalization to unseen recordings would require a recording-disjoint evaluation.
- The saved band-ablation JSON specifies 3,954 training, 848 validation and 51,687 test images. Its membership matches the five existing on-disk scheme partitions.
- The full-data variant and DeepShip use in-memory splits. The files ending in **_reconstructed.csv.gz** describe the membership obtained by replaying their split logic on the currently stored feature filenames. These are reconstructed snapshots. Historical membership for those runs cannot be independently established from the available files. DeepShip's raw source directory was empty during review, and its generator samples from the order returned by the filesystem.
- The main checkpoint's historical report records 0.9584 holdout accuracy, 0.9567 macro F1 and a positive residual scale of 0.4735. The training run logs 0.9528 for its internal test. The cross-experiment summary contains 0.9585 for a historical CTF entry. Keep these records separate and confirm the checkpoint, partition and TTA setting behind each manuscript value.

## Open manuscript and implementation checks

| Finding | Implication |
| --- | --- |
| The research notes mention C-TFDA; the main experiment trains the CTF module. | The CTF code implements learnable chirp rates/frequencies, group-wise chirp convolution, fusion and a gated residual. The separate C-TFDA experiment is in **chirplet/best.py**. Confirm which architecture and name correspond to each manuscript result. |
| The main trainer clamps the learned residual coefficient to be nonnegative. The standalone CTF and SNR evaluators use the coefficient directly. | The stored positive coefficient yields the same effective scale for that checkpoint. A new checkpoint can diverge if the coefficient becomes negative. The release preserves both original formulas. |
| The SNR test adds Gaussian noise to feature-image tensors and samples images by class. | It does not enforce one distinct source recording per sampled image. Cite the implemented feature-domain protocol and verify manuscript descriptions and SNR plots against the tested setup. |
| The PRF comparison exists in the archived results, but a training entrypoint explicitly attributable to that historical PRF run was unavailable. | Preserve the historical evidence and identify the exact source revision before claiming that row can be regenerated from this release. |
| Legacy splits use seeded randomization with unsorted file listings in some scripts. | Use the released manifests for archival ShipsEar splits. Re-running the selection script on a new filesystem can assign different images. |
| The full training software/hardware lock and public checkpoint location are unavailable. | The pinned direct dependencies describe the observed environment. Historical weights remain external, and full training and full-dataset evaluation were not run. |
| No project license or metadata redistribution terms were established from the available files. | Confirm the code license and permission to include the bundled ShipsEar metadata and split lists before public upload. Follow the dataset providers' terms for audio. |

## Release edits and checks

- The source experiment files and historical result files were kept in their original archive. The release retains the experiment code and inline settings, publishes the available memberships, and uses repository-relative paths for local inputs and outputs. Host-specific prefixes in archival result text were normalized. Numeric metrics remain as recorded.
- The original **chirplet/resnet18+CTF_test.py** had an unindented method body. The release fixes its indentation and points its default holdout to the feature folder produced by the documented pipeline. No kernel equations, loss formulas, model structure or historical result values were changed.
- Executable label strings and historical output text are retained, including some Chinese text used in report parsing. Retained source comments are in English. Where the release preparation did not have a reliable translation, the original notes were kept only in the source archive. The source-to-release check verified that comment and docstring editing did not change executable statements.
- Refer to the [PAA submission guidelines](https://link.springer.com/journal/10044/submission-guidelines) when finalizing the title-page declarations and manuscript Data Availability Statement. The [journal scope](https://link.springer.com/journal/10044/aims-and-scope) is separate from its author instructions.

- The split restoration tool passed a synthetic test with all partitions, repeat invocation and stale-output rejection. The main evaluation smoke test loaded the historical checkpoint and identified all five test classes, but inference was terminated with exit 137 in the 2 GiB container. End-to-end evaluation remains unverified.
