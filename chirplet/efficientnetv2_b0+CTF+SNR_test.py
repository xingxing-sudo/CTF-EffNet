import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import timm
import numpy as np
from tqdm import tqdm


class ChirpletTransformFreq(nn.Module):
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
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)
        y = torch.linspace(-1, 1, size, device=freq.device)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        sigma = 0.3
        envelope = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        carrier = torch.cos(2 * np.pi * freq.view(-1, 1, 1) * X +
                            np.pi * chirp_rate.view(-1, 1, 1) * X**2)
        kernel = envelope * carrier
        kernel = kernel / (kernel.norm(p=2, dim=(1,2), keepdim=True) + 1e-8)
        return kernel.float()

    def forward(self, x):
        B, C, H, W = x.shape
        kernels = self._gabor_kernel(self.kernel_size, self.freqs, self.chirp_rates)
        kernels = kernels.unsqueeze(1).repeat(C, 1, 1, 1)
        chirp_feat = F.conv2d(x, kernels, padding=self.kernel_size//2, groups=C)
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

# Full-band additive Gaussian white noise
def add_white_gaussian_noise(x, snr_db):
    if snr_db >= 100:
        return x
    sig_pow = torch.mean(x**2, dim=(1,2,3), keepdim=True)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = torch.randn_like(x) * torch.sqrt(noise_pow)
    noisy = x + noise
    noisy = torch.clamp(noisy, 0.0, 1.0)
    return noisy

# Configuration
TEST_DATA_ROOT = os.environ.get("CTF_TEST_DATA", "./chirplet/chirplet_picture_pro/chirplet_shipsear_small_90_test")
MODEL_PATH = os.environ.get("CTF_CHECKPOINT", "./chirplet/logs/tf_efficientnetv2_b0+CTF/small_10/20260517_114807/best_model.pth")
OUTPUT_DIR = os.path.dirname(os.path.abspath(MODEL_PATH))
NUM_CLASSES = 5
BATCH_SIZE = 64
IMG_SIZE = 224
DEVICE = torch.device("cuda")
SNR_LIST = [100, 20, 10, 5, 0]

np.random.seed(456)  # Seed the sample selection.


transform_test = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
full_dataset = datasets.ImageFolder(TEST_DATA_ROOT, transform_test)


indices = []
for class_idx in range(NUM_CLASSES):
    class_indices = [i for i, label in enumerate(full_dataset.targets) if label == class_idx]
    sampled = np.random.choice(class_indices, 100, replace=False)
    indices.extend(sampled)

subset_dataset = Subset(full_dataset, indices)
test_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

print(f"测试集样本数：{len(subset_dataset)} 张（每类 100 张）")

# Build the model.
def build_backbone():
    backbone = timm.create_model("tf_efficientnetv2_b0", pretrained=False, num_classes=NUM_CLASSES, in_chans=3)
    backbone.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(backbone.num_features, NUM_CLASSES))
    return backbone

backbone = build_backbone()
ctf_module = ChirpletTransformFreq(in_channels=3, n_kernels=32, kernel_size=15)
model = ModelWithCTF(ctf_module, backbone).to(DEVICE)

# Load the checkpoint
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()

print(f"ℹ️ CTF residual_scale = {model.ctf.residual_scale.item():.4f}")


@torch.no_grad()
def test(model, loader, snr):
    preds, labels = [], []
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    for x, y in tqdm(loader, desc=f"SNR={snr}dB"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x = add_white_gaussian_noise(x, snr)
        x = (x - mean) / std
        out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())

    return np.concatenate(labels), np.concatenate(preds)


print("\n========== CTF 模型噪声鲁棒性测试（标准高斯白噪声，独立样本） ==========\n")
acc_results = {}
for s in SNR_LIST:
    l, p = test(model, test_loader, s)
    acc = (l == p).mean()
    acc_results[s] = acc
    print(f"SNR={s:3d} dB | Acc = {acc:.4f}")

# Save results.
result_file = os.path.join(OUTPUT_DIR, "noise_robustness_CTF_only.txt")
with open(result_file, "w") as f:
    f.write("SNR(dB)\tAccuracy\n")
    for s in SNR_LIST:
        f.write(f"{s if s != 100 else 'Clean'}\t{acc_results[s]:.4f}\n")

print(f"\n✅ 实验完成！结果已保存至: {result_file}")