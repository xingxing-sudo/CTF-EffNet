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


class CTFModule(nn.Module):
    """CTF module used by the standalone evaluator."""
    def __init__(self, in_channels=3, n_kernels=32, kernel_size=15):
        super().__init__()
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # Learnable chirp rates and frequencies
        chirp_rates = torch.linspace(-5, 5, n_kernels)
        freqs = torch.linspace(0.1, 0.9, n_kernels)
        
        self.chirp_rates = nn.Parameter(chirp_rates.float()) 
        self.freqs = nn.Parameter(freqs.float())

        # Batch normalization after convolution
        self.bn_after_conv = nn.BatchNorm2d(in_channels * n_kernels)


        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * n_kernels, in_channels * 8, 1, bias=False),
            nn.BatchNorm2d(in_channels * 8),
            nn.GELU(),
            nn.Conv2d(in_channels * 8, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)
        y = torch.linspace(-1, 1, size, device=freq.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(
            2 * np.pi * freq.view(-1, 1, 1) * X + 
            np.pi * chirp_rate.view(-1, 1, 1) * X**2
        )
        kernel = envelope * carrier
        kernel = kernel / (kernel.norm(p=2, dim=(1,2), keepdim=True) + 1e-8)
        return kernel.float()

    def forward(self, x):
        B, C, H, W = x.shape
        kernels = self._gabor_kernel(self.kernel_size, self.freqs, self.chirp_rates)
        kernels = kernels.unsqueeze(1).repeat(C, 1, 1, 1)
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

# Configuration

TEST_DATA_ROOT = os.environ.get("CTF_TEST_DATA", "./chirplet/chirplet_picture_pro/chirplet_shipsear_small_90_test")
MODEL_PATH = os.environ.get("CTF_CHECKPOINT", "./chirplet/logs/tf_efficientnetv2_b0+CTF/small_10/20260517_114807/best_model.pth")

OUTPUT_DIR = os.path.dirname(os.path.abspath(MODEL_PATH))
NUM_CLASSES = 5
CLASS_NAMES = [
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise"
]
SHORT_NAMES = ["A", "B", "C", "D", "E"]

BATCH_SIZE = int(os.environ.get("CTF_BATCH_SIZE", "128"))
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
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=int(os.environ.get("CTF_WORKERS", "8")), pin_memory=True)
print(f"测试集样本数：{len(test_dataset)} 张")

# Build the model.
backbone = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=NUM_CLASSES,
    in_chans=3,
    drop_rate=0.2,
    drop_path_rate=0.1,
)
num_ftrs = backbone.num_features
backbone.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, NUM_CLASSES)
)

# Instantiate the CTF module.
ctf_module = CTFModule(in_channels=3, n_kernels=32, kernel_size=15)
model = ModelWithCTF(ctf_module, backbone).to(DEVICE)

# Load the checkpoint
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    print("ℹ️ 已去除 '_orig_mod.' 前缀")

model.load_state_dict(state_dict, strict=True) 
model.eval()

# Inspect the CTF residual scale.
print(f"✅ 模型权重加载成功：{MODEL_PATH}")
print(f"ℹ️ CTF 残差缩放系数 (Residual Scale): {model.ctf.residual_scale.item():.4f}")

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
print(f"CTF 模型测试集准确率 (TTA={USE_TTA}): {accuracy:.4f}")
print("=" * 60)
print(report)

# Save results.
result_file = os.path.join(OUTPUT_DIR, f"test_results_tta{USE_TTA}.txt")
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"模型路径: {MODEL_PATH}\n")
    f.write(f"CTF Residual Scale: {model.ctf.residual_scale.item():.4f}\n")
    f.write(f"准确率: {accuracy:.4f}\n\n")
    f.write(report)
print(f"📄 报告已保存: {result_file}")

# Plot the confusion matrix.
plt.figure(figsize=(10, 8))
cm_sum = cm.sum(axis=1, keepdims=True)
cm_perc = cm / cm_sum * 100
annot = np.empty_like(cm, dtype='object')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
plt.title(f'Confusion Matrix (Acc={accuracy:.4f})')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, f"test_confusion_matrix_tta{USE_TTA}.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"📊 混淆矩阵已保存: {cm_path}")