import os
import shutil

# Input and output paths
SOURCE_DIR = "./wst/wst_picture/all_wst_picture"  # Root directory of the generated features
FINAL_TEST_DIR = "./wst/wst_picture/wst_shipsear_small_90_test"  # Output directory for the held-out test set
SMALL_SAMPLE_FILE_LIST = 'wst/picture_split/small_sample_file_list.txt'

# Read the list of small-sample filenames.
with open(SMALL_SAMPLE_FILE_LIST, "r") as f:
    small_sample_files = set([line.strip() for line in f.readlines()])

# Create the held-out test directory.
os.makedirs(FINAL_TEST_DIR, exist_ok=True)

# Copy feature images outside the small-sample partition.
for cls in os.listdir(SOURCE_DIR):
    cls_src_dir = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_src_dir):
        continue
    # Create the class directory.
    cls_dst_dir = os.path.join(FINAL_TEST_DIR, cls)
    os.makedirs(cls_dst_dir, exist_ok=True)
    # Keep images outside the small-sample list.
    for img in os.listdir(cls_src_dir):
        if img.endswith(".png") and img not in small_sample_files:
            src_path = os.path.join(cls_src_dir, img)
            dst_path = os.path.join(cls_dst_dir, img)
            shutil.copy2(src_path, dst_path)
    print(f"{cls}：终极测试集已复制{len(os.listdir(cls_dst_dir))}张图片")

print(f"\n终极测试集创建完成，目录：{FINAL_TEST_DIR}")
print(f"该测试集包含原始数据的90%左右，未参与任何小样本训练/验证")