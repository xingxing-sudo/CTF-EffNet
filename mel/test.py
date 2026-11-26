import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


TEST_DATA_ROOT = os.environ.get("CTF_TEST_DATA", "./mel/mel_picture/mel_shipsear_small_90_test")
MODEL_PATH = os.environ.get("CTF_CHECKPOINT", "./mel/logs/tf_efficientnetv2_b0/small_10/20260429_095657/best_model.pth")
OUTPUT_DIR = os.path.dirname(os.path.abspath(MODEL_PATH))
NUM_CLASSES = 5
CLASS_NAMES = [
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise"
]
BATCH_SIZE = 128
IMG_SIZE = 224
MODEL_NAME = "tf_efficientnetv2_b0"
USE_TTA = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test set
print(f"📂 加载测试集：{TEST_DATA_ROOT}")
test_dataset = datasets.ImageFolder(TEST_DATA_ROOT, transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
print(f"测试集样本数：{len(test_dataset)} 张")
print(f"类别分布：{dict(zip(CLASS_NAMES, [test_dataset.targets.count(i) for i in range(NUM_CLASSES)]))}")

# Load the model.
model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    in_chans=3,
    drop_rate=0.2,
    drop_path_rate=0.1,
)
num_ftrs = model.num_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

# Strip the torch.compile prefix from checkpoint keys where present.
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    print("ℹ️ 已去除 '_orig_mod.' 前缀")
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print(f"✅ 模型已加载：{MODEL_PATH}")

# Evaluation
@torch.no_grad()
def evaluate(loader, use_tta=False):
    all_preds = []
    all_labels = []
    
    for x, y in tqdm(loader, desc="Testing"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        if use_tta:
            pred_probs = torch.softmax(model(x), dim=1)
            for shift in [-2, 2]:
                x_shifted = torch.roll(x, shifts=shift, dims=2)
                pred_probs += torch.softmax(model(x_shifted), dim=1)
            pred_probs /= 3.0
            preds = pred_probs.argmax(1)
        else:
            outputs = model(x)
            preds = outputs.argmax(1)
            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    return np.array(all_labels), np.array(all_preds)

# Run evaluation
print("\n🧪 开始测试评估...")
true_labels, pred_labels = evaluate(test_loader, use_tta=USE_TTA)

# Calculate metrics.
accuracy = (true_labels == pred_labels).mean()
report = classification_report(true_labels, pred_labels, target_names=CLASS_NAMES, digits=4)
cm = confusion_matrix(true_labels, pred_labels)

print("\n" + "=" * 60)
print(f"测试集准确率 (TTA={USE_TTA}): {accuracy:.4f}")
print("=" * 60)
print(report)

# Write results to disk.
result_file = os.path.join(OUTPUT_DIR, f"test_results_tta{USE_TTA}.txt")
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"模型路径: {MODEL_PATH}\n")
    f.write(f"测试集路径: {TEST_DATA_ROOT}\n")
    f.write(f"测试集样本数: {len(test_dataset)}\n")
    f.write(f"TTA: {USE_TTA}\n")
    f.write("=" * 60 + "\n")
    f.write(f"准确率: {accuracy:.4f}\n\n")
    f.write(report)
print(f"\n📄 分类报告已保存至: {result_file}")

# Plot the confusion matrix.
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix (TTA={USE_TTA}, Acc={accuracy:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, f"test_confusion_matrix_tta{USE_TTA}.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"📊 混淆矩阵已保存至: {cm_path}")