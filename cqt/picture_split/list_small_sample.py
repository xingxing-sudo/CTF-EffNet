# List feature image names in the small-sample partition.
import os

# Root directory of the small-sample partition
SMALL_DATASET_DIR = "./cqt/cqt_picture/cqt_shipsear_small_10_train"
# Collect small-sample filenames
small_sample_files = set()

# Collect the filenames from all three small-sample subsets.
for split in ["train", "val", "test"]:
    split_dir = os.path.join(SMALL_DATASET_DIR, split)
    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        for img in os.listdir(cls_dir):
            if img.endswith(".png"):
                small_sample_files.add(img)  # Deduplicate filenames with a set.

# Write the list to disk.
with open('cqt/picture_split/small_sample_file_list.txt', "w") as f:
    for img in small_sample_files:
        f.write(f"{img}\n")
print(f"小样本文件名清单已保存，共{len(small_sample_files)}张图片")