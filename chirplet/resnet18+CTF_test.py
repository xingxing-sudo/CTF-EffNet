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

# Configuration
TEST_DATA_ROOT = os.environ.get("CTF_TEST_DATA", "./chirplet/chirplet_picture_pro/chirplet_shipsear_small_90_test")
MODEL_PATH = os.environ.get("CTF_CHECKPOINT", "./chirplet/logs/resnet18+CTF/small_10/20260506_130316/best_model.pth")
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
MODEL_NAME = "resnet18"
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


class CTFModule(nn.Module):
    def __init__(self, in_channels=3, n_kernels=32, kernel_size=15):
        super().__init__()
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        chirp_rates = torch.linspace(-5, 5, n_kernels)
        freqs = torch.linspace(0.1, 0.9, n_kernels)
        
        self.chirp_rates = nn.Parameter(chirp_rates.float()) 
        self.freqs = nn.Parameter(freqs.float())

        self.bn_after_conv = nn.BatchNorm2d(in_channels * n_kernels)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * n_kernels, in_channels * 8, 1, bias=False),
            nn.BatchNorm2d(in_channels * 8),
            nn.GELU(),
            nn.Conv2d(in_channels * 8, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)  # Time
        y = torch.linspace(-1, 1, size, device=freq.device)  # Frequency
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(2 * np.pi * freq.view(-1,1,1) * X + 
                            np.pi * chirp_rate.view(-1,1,1) * X**2)
        kernel = envelope * carrier
        kernel = kernel / (kernel.norm(p=2, dim=(1,2), keepdim=True) + 1e-8)
        return kernel.float()

    def forward(self, x):
        B, C, H, W = x.shape
        
        kernels = self._gabor_kernel(self.kernel_size, self.freqs, self.chirp_rates)
        kernels = kernels.unsqueeze(1)
        kernels = kernels.repeat(C, 1, 1, 1)
        
        chirp_feat = nn.functional.conv2d(x, kernels, padding=self.kernel_size//2, groups=C)
        chirp_feat = self.bn_after_conv(chirp_feat)
        
        residual = self.fusion(chirp_feat)
        
        return x + self.residual_scale * residual

# Model wrapper
class ModelWithCTF(nn.Module):
    def __init__(self, ctf, backbone):
        super().__init__()
        self.ctf = ctf
        self.backbone = backbone
    def forward(self, x):
        x = self.ctf(x)
        return self.backbone(x)


backbone = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    in_chans=3,
    drop_rate=0.2,
)

num_ftrs = backbone.num_features
backbone.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

ctf_module = CTFModule(in_channels=3, n_kernels=32, kernel_size=15)
model = ModelWithCTF(ctf_module, backbone)

# Load the checkpoint
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    print("ℹ️ 已去除 '_orig_mod.' 前缀")

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print(f"✅ ResNet18 + CTF 模型加载成功：{MODEL_PATH}")


ctf_scale = model.ctf.residual_scale.item()
print(f"📊 最终 CTF Residual Scale: {ctf_scale:.4f}")

if abs(ctf_scale) < 1e-3:
    print("⚠️ 警告：CTF Residual Scale 接近 0，模块可能未被有效激活！")
    print("   建议检查训练日志，确认训练过程中该值是否逐渐上升。")

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


accuracy = (true_labels == pred_labels).mean()
report = classification_report(true_labels, pred_labels, target_names=CLASS_NAMES, digits=4)
cm = confusion_matrix(true_labels, pred_labels)

print("\n" + "=" * 60)
print(f"测试集准确率 (TTA={USE_TTA}): {accuracy:.4f}")
print("=" * 60)
print(report)

# Save results.
result_file = os.path.join(OUTPUT_DIR, f"test_results_tta{USE_TTA}.txt")
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"模型: ResNet18 + CTF\n")
    f.write(f"最终 CTF Scale: {ctf_scale:.4f}\n")
    f.write(f"准确率: {accuracy:.4f}\n\n")
    f.write(report)

# Confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title(f'ResNet18 + CTF Confusion Matrix (Acc={accuracy:.4f})')
plt.ylabel('True')
plt.xlabel('Pred')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.pdf"), dpi=300, bbox_inches='tight')

print("\n🎉 测试完成！结果保存在：")
print(f"  - 测试报告：{result_file}")
print(f"  - 混淆矩阵：{os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}")