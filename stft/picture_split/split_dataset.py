import os
import shutil
import random

# Input and output paths
SOURCE_DIR = "./stft/stft_picture/all_stft_picture"  # Root directory of the generated features
OUTPUT_DIR = "./stft/stft_picture/stft_shipsear_small_10_train"  # Output directory for split images


# Seed Python's random generator.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create output directories
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print("=" * 60)
print("数据集重新划分工具 (训练/验证/测试)")
print("目标: 解决类别不平衡问题")
print("=" * 60)

# Count the input samples
print("\n📊 原始数据统计:")
print("-" * 40)

class_stats = {}

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if os.path.isdir(cls_path):
        # Count PNG feature images.
        images = [f for f in os.listdir(cls_path) if f.endswith('.png')]
        class_stats[cls] = len(images)
        print(f"{cls}: {len(images)}张")

# Split protocol
print("\n🎯 重新划分策略 (7:1.5:1.5):")
print("-" * 40)
print("训练集: 70%, 验证集: 15%, 测试集: 15%")
print("C_Fishing_Boat: 增加验证和测试集比例（65%:17.5%:17.5%）")

# Create the dataset split
total_train = 0
total_val = 0
total_test = 0

for cls, count in class_stats.items():
    cls_src_path = os.path.join(SOURCE_DIR, cls)
    # List the PNG files for this class.
    images = [f for f in os.listdir(cls_src_path) if f.endswith('.png')]
    random.shuffle(images)  # Shuffle the image filenames.

    SAMPLE_RATIO = 0.1  # Fraction sampled before creating subsets
    random.shuffle(images)  # Shuffle before selecting samples.
    sample_size = int(len(images) * SAMPLE_RATIO)
    images = images[:sample_size]  # Keep the selected prefix of the shuffled list.
    print(f"{cls}：原始{count}张 → 小样本总{len(images)}张（保留{SAMPLE_RATIO*100:.1f}%）")
    # ===========================================================
    # Choose the class-specific split ratios.
    if cls == "C_Fishing_Boat":
        # Fishing boat split thresholds are 65% and 17.5%, with the remainder for testing.
        train_ratio = 0.65
        val_ratio = 0.175
    else:
        # Default split thresholds are 70% and 15%, with the remainder for testing.
        train_ratio = 0.70
        val_ratio = 0.15
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Calculate the split boundaries.
    train_end = int(len(images) * train_ratio)
    val_end = train_end + int(len(images) * val_ratio)
    
    # Partition the selected feature images.
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    # Create the output directory for this class in each partition.
    train_cls_dst = os.path.join(TRAIN_DIR, cls)
    val_cls_dst = os.path.join(VAL_DIR, cls)
    test_cls_dst = os.path.join(TEST_DIR, cls)
    os.makedirs(train_cls_dst, exist_ok=True)
    os.makedirs(val_cls_dst, exist_ok=True)
    os.makedirs(test_cls_dst, exist_ok=True)
    

    for img in train_images:
        src = os.path.join(cls_src_path, img)
        dst = os.path.join(train_cls_dst, img)
        shutil.copy2(src, dst)
    
    for img in val_images:
        src = os.path.join(cls_src_path, img)
        dst = os.path.join(val_cls_dst, img)
        shutil.copy2(src, dst)
    
    for img in test_images:
        src = os.path.join(cls_src_path, img)
        dst = os.path.join(test_cls_dst, img)
        shutil.copy2(src, dst)
    
    # Update total counts.
    total_train += len(train_images)
    total_val += len(val_images)
    total_test += len(test_images)
    
    # Print the counts for this class.
    print(f"\n{cls}:")
    print(f"  训练集: {len(train_images)}张 ({len(train_images)/count*100:.1f}%)")
    print(f"  验证集: {len(val_images)}张 ({len(val_images)/count*100:.1f}%)")
    print(f"  测试集: {len(test_images)}张 ({len(test_images)/count*100:.1f}%)")

# Check the generated partitions.
print("\n✅ 划分完成！最终统计:")
print("=" * 60)

# Count training examples.
print(f"\n训练集 ({TRAIN_DIR}):")
train_stats = {}
for cls in os.listdir(TRAIN_DIR):
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.isdir(cls_path):
        img_count = len([f for f in os.listdir(cls_path) if f.endswith('.png')])
        train_stats[cls] = img_count
        print(f"  {cls}: {img_count}张")

# Count validation examples.
print(f"\n验证集 ({VAL_DIR}):")
val_stats = {}
for cls in os.listdir(VAL_DIR):
    cls_path = os.path.join(VAL_DIR, cls)
    if os.path.isdir(cls_path):
        img_count = len([f for f in os.listdir(cls_path) if f.endswith('.png')])
        val_stats[cls] = img_count
        print(f"  {cls}: {img_count}张")

# Count test examples.
print(f"\n测试集 ({TEST_DIR}):")
test_stats = {}
for cls in os.listdir(TEST_DIR):
    cls_path = os.path.join(TEST_DIR, cls)
    if os.path.isdir(cls_path):
        img_count = len([f for f in os.listdir(cls_path) if f.endswith('.png')])
        test_stats[cls] = img_count
        print(f"  {cls}: {img_count}张")

# Aggregate counts
print(f"\n📈 总计:")
print(f"  训练集: {total_train}张")
print(f"  验证集: {total_val}张")
print(f"  测试集: {total_test}张")
print(f"  总数据: {total_train + total_val + total_test}张")
print(f"  比例: {total_train}:{total_val}:{total_test} (近似7:1.5:1.5)")

# Check class proportions.
print(f"\n⚖️ 分布平衡性检查:")
print("-" * 40)
for cls in class_stats.keys():
    train_pct = (train_stats.get(cls, 0) / total_train * 100) if total_train > 0 else 0.0
    val_pct = (val_stats.get(cls, 0) / total_val * 100) if total_val > 0 else 0.0
    test_pct = (test_stats.get(cls, 0) / total_test * 100) if total_test > 0 else 0.0
    
    print(f"\n{cls}:")
    print(f"  训练集占比: {train_pct:.1f}%")
    print(f"  验证集占比: {val_pct:.1f}%")
    print(f"  测试集占比: {test_pct:.1f}%")

# Usage instructions
print(f"\n🚀 使用说明:")
print("=" * 60)
print(f"1. 新的平衡数据集在: {OUTPUT_DIR}")
print(f"2. 训练命令 (带验证集):")
print(f"   python main.py \\")
print(f"     --data_set image_folder \\")
print(f"     --data_path {TRAIN_DIR} \\")
print(f"     --eval_data_path {TEST_DIR} \\")
print(f"     --nb_classes 5 \\")
print(f"     --model convnext_small")

print(f"\n3. 如果要使用验证集进行早停:")
print(f"   python main.py \\")
print(f"     --data_set image_folder \\")
print(f"     --data_path {TRAIN_DIR} \\")
print(f"     --eval_data_path {VAL_DIR} \\")  # Use the validation set for model selection.
print(f"     --nb_classes 5")

print(f"\n4. 目录结构:")
print(f"   {OUTPUT_DIR}/")
print(f"   ├── train/       # 训练集（约70%）")
print(f"   ├── val/         # 验证集（约15%）")
print(f"   └── test/        # 测试集（约15%）")

print(f"\n5. 关键改进:")
print("   - C_Fishing_Boat测试集占比提升至17.5%，解决类别不平衡问题")
print("   - 验证集和测试集保留足够样本，避免数据泄露")
print("   - 随机打乱后划分，保证数据分布均匀")
print("   - 保留原始文件元信息，不损坏特征图数据")