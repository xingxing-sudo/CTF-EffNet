import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
import time



class CTFModule_Ablation1_NoChirp(nn.Module):
    def __init__(self, in_channels=3, n_kernels=32, kernel_size=15):
        super().__init__()
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        freqs = torch.linspace(0.1, 0.9, n_kernels)
        self.freqs = nn.Parameter(freqs.float())
        self.chirp_rates = torch.zeros(n_kernels)  # Keep the parameter fixed at zero.
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
        x = torch.linspace(-1, 1, size, device=freq.device)  # Time axis
        y = torch.linspace(-1, 1, size, device=freq.device)  # Frequency axis
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(2 * np.pi * freq.view(-1, 1, 1) * X)  # Zero chirp rate
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



class CTFModule_Ablation2_NoFusion(nn.Module):
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
        self.simple_fusion = nn.Sequential(
            nn.Conv2d(in_channels * n_kernels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        nn.init.zeros_(self.simple_fusion[-1].weight)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)
        y = torch.linspace(-1, 1, size, device=freq.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(2 * np.pi * freq.view(-1, 1, 1) * X + np.pi * chirp_rate.view(-1, 1, 1) * X**2)
        kernel = envelope * carrier
        kernel = kernel / (kernel.norm(p=2, dim=(1,2), keepdim=True) + 1e-8)
        return kernel.float()

    def forward(self, x):
        B, C, H, W = x.shape
        kernels = self._gabor_kernel(self.kernel_size, self.freqs, self.chirp_rates)
        kernels = kernels.unsqueeze(1).repeat(C, 1, 1, 1)
        chirp_feat = nn.functional.conv2d(x, kernels, padding=self.kernel_size//2, groups=C)
        chirp_feat = self.bn_after_conv(chirp_feat)
        residual = self.simple_fusion(chirp_feat)
        return x + self.residual_scale * residual



class CTFModule_Ablation3_NoGate(nn.Module):
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


    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)
        y = torch.linspace(-1, 1, size, device=freq.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(2 * np.pi * freq.view(-1, 1, 1) * X + np.pi * chirp_rate.view(-1, 1, 1) * X**2)
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
        return x + residual  # Ungated residual connection


# Model wrapper
class ModelWithCTF(nn.Module):
    def __init__(self, ctf, backbone):
        super().__init__()
        self.ctf = ctf
        self.backbone = backbone

    def forward(self, x):
        x = self.ctf(x)
        return self.backbone(x)



EXPERIMENTS = [
    {
        "name": "w/o Learnable Chirp Prior",
        "short_name": "No_Chirp_Prior",
        "model_path": "./chirplet/logs/ctf_ablation/tf_efficientnetv2_b0+CTF_Ablation1_NoChirpPrior/small_10/20260516_201416/best_model.pth",
        "ctf_class": CTFModule_Ablation1_NoChirp
    },
    {
        "name": "w/o Deep Fusion",
        "short_name": "No_Fusion",
        "model_path": "./chirplet/logs/ctf_ablation/tf_efficientnetv2_b0+CTF_Ablation2_NoFusion/small_10/20260517_114835/best_model.pth",
        "ctf_class": CTFModule_Ablation2_NoFusion
    },
    {
        "name": "w/o Residual Gate",
        "short_name": "No_Gate",
        "model_path": "./chirplet/logs/ctf_ablation/tf_efficientnetv2_b0+CTF_Ablation3_NoGate/small_10/20260516_112102/best_model.pth",
        "ctf_class": CTFModule_Ablation3_NoGate
    }
]


# Global configuration
TEST_DATA_ROOT = "./chirplet/chirplet_picture_pro/chirplet_shipsear_small_90_test"
OUTPUT_ROOT = "./chirplet/logs/ctf_ablation/test_results_1_FULL"
NUM_CLASSES = 5
CLASS_NAMES = ["A_Large_Merchant_Ship","B_Small_Speedboat","C_Fishing_Boat","D_Ferry","E_Ambient_Noise"]
SHORT_NAMES = ["A", "B", "C", "D", "E"]
BATCH_SIZE = 128
IMG_SIZE = 224
MODEL_NAME = "tf_efficientnetv2_b0"
USE_TTA = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Data preprocessing
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test set
test_dataset = datasets.ImageFolder(TEST_DATA_ROOT, transform_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# Evaluation
@torch.no_grad()
def evaluate_model(model, loader, use_tta=False):
    all_preds, all_labels = [], []
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
            preds = model(x).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

# Evaluation workflow
all_results = []
for exp in EXPERIMENTS:
    print("="*70 + f"\n🧪 测试：{exp['name']}\n" + "="*70)
    exp_output_dir = os.path.join(OUTPUT_ROOT, exp['short_name'])
    os.makedirs(exp_output_dir, exist_ok=True)

    # Build the model.
    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, in_chans=3)
    backbone.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(backbone.num_features, NUM_CLASSES))
    ctf_module = exp['ctf_class'](in_channels=3, n_kernels=32, kernel_size=15)
    model = ModelWithCTF(ctf_module, backbone).to(DEVICE)
    model.eval()

    # Load the checkpoint
    state_dict = torch.load(exp['model_path'], map_location=DEVICE, weights_only=True)
    state_dict = {k.replace('_orig_mod.', '').replace('prf.', 'ctf.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    # Run evaluation
    true_labels, pred_labels = evaluate_model(model, test_loader, use_tta=USE_TTA)
    accuracy = (true_labels == pred_labels).mean()
    kappa = cohen_kappa_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=CLASS_NAMES, digits=4, output_dict=True)
    
    # Write results.
    print(f"✅ 准确率：{accuracy:.4f} | Kappa：{kappa:.4f}")
    all_results.append({
        "Config": exp['name'], "Accuracy": accuracy, "Kappa": kappa,
        "Macro-P": report['macro avg']['precision'],
        "Macro-R": report['macro avg']['recall'],
        "Macro-F1": report['macro avg']['f1-score']
    })

# Save the summary table.
df = pd.DataFrame(all_results).round(4)
df.to_csv(os.path.join(OUTPUT_ROOT, "ablation_3models_summary.csv"), index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("🎉 3个消融模型测试完成！")
print(f"📊 汇总结果已保存至：{OUTPUT_ROOT}/ablation_3models_summary.csv")
print("="*70)