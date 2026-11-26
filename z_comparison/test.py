import os


BASE_PATH = "./chirplet/chirplet_picture_pro/all_chirplet_picture"


CLASSES = [
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise"
]

print("=" * 60)
print("📊 ShipsEar 数据集 5 类样本数量统计")
print("=" * 60)

class_counts = {}
total_samples = 0

for cls in CLASSES:
    cls_path = os.path.join(BASE_PATH, cls)
    if not os.path.exists(cls_path):
        print(f"❌ 路径不存在：{cls}")
        continue
    

    count = sum(
        1 for f in os.listdir(cls_path)
        if os.path.isfile(os.path.join(cls_path, f))
    )
    class_counts[cls] = count
    total_samples += count
    print(f"✅ {cls:25s} : {count:,} 张")

print("-" * 60)
print(f"📈 总计样本数：{total_samples:,} 张")
print("=" * 60)


print("\n📝 论文表1格式：")
print("| 类别 | 样本数 |")
print("|------|--------|")
for cls, cnt in class_counts.items():
    print(f"| {cls} | {cnt:,} |")
print(f"| **总计** | **{total_samples:,}** |")