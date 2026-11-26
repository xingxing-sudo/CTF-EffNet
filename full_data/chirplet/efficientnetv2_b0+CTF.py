import os
os.environ["OMP_NUM_THREADS"] = "1"

import timm
import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import time
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Import optional GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITOR = True
except:
    GPU_MONITOR = False
    print("⚠️ pynvml 未安装，跳过 GPU 显存监控")

# Experiment configuration
DATA_ROOT = "./chirplet/chirplet_picture_pro/all_chirplet_picture"
NUM_CLASSES = 5
CLASS_NAMES = [
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise"
]
RANDOM_SEED = 456

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-4
IMG_SIZE = 224
MODEL_NAME = "tf_efficientnetv2_b0"

# MixUp settings
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Log directory
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_ROOT = "./full_data/chirplet"
LOG_DIR = os.path.join(LOG_ROOT, MODEL_NAME + "+CTF_FULL_DATA", "7_1.5_1.5_split", time_str)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "training_log.txt")
PLOT_PATH = os.path.join(LOG_DIR, "metrics.png")
MODEL_PATH = os.path.join(LOG_DIR, "best_model.pth")
VIS_SAVE_PATH = os.path.join(LOG_DIR, "ctf_vis_samples")
os.makedirs(VIS_SAVE_PATH, exist_ok=True)

# Set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)

# Chirplet spectrogram augmentation
class SpecAugmentChirplet:
    def __init__(self, freq_mask=12, time_mask=18, channel_drop_prob=0.2, prob=0.6):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.channel_drop_prob = channel_drop_prob
        self.prob = prob

    def __call__(self, x):
        if torch.rand(1) < 0.5:
            gain = torch.empty(1).uniform_(0.6, 1.5).item()
            x = x * gain

        if torch.rand(1) < self.prob:
            f = torch.randint(0, self.freq_mask, (1,)).item()
            if x.shape[1] > f:
                f0 = torch.randint(0, x.shape[1] - f, (1,)).item()
                x[:, f0:f0 + f, :] = 0.0
            t = torch.randint(0, self.time_mask, (1,)).item()
            if x.shape[2] > t:
                t0 = torch.randint(0, x.shape[2] - t, (1,)).item()
                x[:, :, t0:t0 + t] = 0.0

        if torch.rand(1) < self.channel_drop_prob:
            c = torch.randint(0, 3, (1,)).item()
            x[c, :, :] = 0.0

        return x

# Data preprocessing
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(3),
    transforms.ToTensor(),
    SpecAugmentChirplet(freq_mask=12, time_mask=18, channel_drop_prob=0.2, prob=0.6),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print("=" * 70)
print(f"📂 加载完整数据集并自动划分 7:1.5:1.5 | 路径：{DATA_ROOT}")

full_dataset = datasets.ImageFolder(DATA_ROOT, transform=None)
all_paths = [s[0] for s in full_dataset.samples]
all_labels = full_dataset.targets

def get_audio_base_name(path):
    fname = os.path.splitext(os.path.basename(path))[0]
    return "_".join(fname.split("_")[:-1])

group_dict = {}
for idx, path in enumerate(all_paths):
    group = get_audio_base_name(path)
    if group not in group_dict:
        group_dict[group] = []
    group_dict[group].append(idx)

train_indices = []
val_indices = []
test_indices = []

for group, indices in group_dict.items():
    if len(indices) < 3:
        train_indices.extend(indices)
        continue
    train, temp = train_test_split(indices, test_size=0.3, random_state=RANDOM_SEED)
    val, test = train_test_split(temp, test_size=0.5, random_state=RANDOM_SEED)
    train_indices.extend(train)
    val_indices.extend(val)
    test_indices.extend(test)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)
test_dataset = Subset(full_dataset, test_indices)

class DatasetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = DatasetWithTransform(train_dataset, transform_train)
val_dataset = DatasetWithTransform(val_dataset, transform_val)
test_dataset = DatasetWithTransform(test_dataset, transform_val)

print(f"✅ 数据集划分完成（7:1.5:1.5）")
print(f"训练集：{len(train_dataset)} 张 (70%)")
print(f"验证集：{len(val_dataset)} 张 (15%)")
print(f"测试集：{len(test_dataset)} 张 (15%)")
print("=" * 70)


def count_classes_from_indices(indices, full_labels, num_classes):
    cnt = [0] * num_classes
    for idx in indices:
        lbl = full_labels[idx]
        cnt[lbl] += 1
    return cnt

train_cnt = count_classes_from_indices(train_indices, all_labels, NUM_CLASSES)
val_cnt = count_classes_from_indices(val_indices, all_labels, NUM_CLASSES)
test_cnt = count_classes_from_indices(test_indices, all_labels, NUM_CLASSES)

print("\n📊 各类别数量统计：")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {name}: 训练 {train_cnt[i]} | 验证 {val_cnt[i]} | 测试 {test_cnt[i]}")

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE * 2, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE * 2, shuffle=False, num_workers=8, pin_memory=True)


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
        if self.fusion[-1].bias is not None:
            nn.init.zeros_(self.fusion[-1].bias)
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def _gabor_kernel(self, size, freq, chirp_rate):
        x = torch.linspace(-1, 1, size, device=freq.device)
        y = torch.linspace(-1, 1, size, device=freq.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        envelope = torch.exp(- (X**2 + Y**2) / 0.3)
        carrier = torch.cos(2 * np.pi * freq.view(-1, 1, 1) * Y + 
                            np.pi * chirp_rate.view(-1, 1, 1) * Y**2 + 
                            2 * np.pi * freq.view(-1, 1, 1) * X)
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

# Model definition
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

class ModelWithCTF(nn.Module):
    def __init__(self, ctf, backbone):
        super().__init__()
        self.ctf = ctf
        self.backbone = backbone
    def forward(self, x):
        x = self.ctf(x)
        return self.backbone(x)

ctf_module = CTFModule(in_channels=3, n_kernels=32, kernel_size=15)
model = ModelWithCTF(ctf_module, backbone).to(DEVICE)

try:
    model = torch.compile(model)
    print("✅ Torch compile 已启用")
except:
    print("⚠️ Torch compile 不可用，跳过")

# ==================== MixUp ====================
from timm.data.mixup import Mixup
mixup_fn = Mixup(
    mixup_alpha=MIXUP_ALPHA,
    cutmix_alpha=CUTMIX_ALPHA,
    prob=MIXUP_PROB,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=NUM_CLASSES
)

# Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
warmup_epochs = 10
scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
scheduler2 = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

scaler = torch.amp.GradScaler('cuda')
GRAD_CLIP_NORM = 1.0

# Helper functions
def per_class_accuracy(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    return acc

def save_ctf_visualization(model, val_loader, save_dir, device, epoch):
    model.eval()
    data_iter = iter(val_loader)
    x, _ = next(data_iter)
    x = x.to(device)
    
    with torch.no_grad():
        x_ctf = model.ctf(x)
    
    def normalize_for_plot(img_tensor):
        img_np = img_tensor.cpu().numpy()
        vmin = img_np.min()
        vmax = img_np.max()
        img_np = (img_np - vmin) / (vmax - vmin + 1e-8)
        return np.transpose(img_np, (1, 2, 0))

    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    for i in range(4):
        axes[i, 0].imshow(normalize_for_plot(x[i]))
        axes[i, 0].set_title(f"Input (Epoch {epoch})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(normalize_for_plot(x_ctf[i]))
        scale_val = model.ctf.residual_scale.item()
        axes[i, 1].set_title(f"After CTF (Scale={scale_val:.3f})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ctf_compare_epoch_{epoch}.png"), dpi=100)
    plt.close()

# Training
def train_one_epoch():
    model.train()
    loss_sum = 0.0
    corr_sum = 0
    tot = 0

    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            pred = model(x)
            lo = criterion(pred, y)

        scaler.scale(lo).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += lo.item() * x.size(0)
        if len(y.shape) == 2:
            pred_label = pred.argmax(1)
            true_label = y.argmax(1)
        else:
            pred_label = pred.argmax(1)
            true_label = y
        corr_sum += (pred_label == true_label).sum().item()
        tot += x.size(0)

    return loss_sum / tot, corr_sum / tot

def evaluate(loader, use_tta=False):
    model.eval()
    loss_sum = 0.0
    corr_sum = 0
    tot = 0
    preds, labs = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)

            if use_tta:
                pred_probs = torch.softmax(model(x), dim=1)
                for shift in [-2, 2]:
                    x_shifted = torch.roll(x, shifts=shift, dims=2)
                    pred_probs += torch.softmax(model(x_shifted), dim=1)
                pred_probs /= 3.0
                pred = pred_probs
                lo = criterion(pred, y)
            else:
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    lo = criterion(pred, y)

            loss_sum += lo.item() * x.size(0)
            final_pred = pred.argmax(1) if not use_tta else pred_probs.argmax(1)
            corr_sum += (final_pred == y).sum().item()
            tot += x.size(0)
            preds.extend(final_pred.cpu().numpy())
            labs.extend(y.cpu().numpy())

    return loss_sum / tot, corr_sum / tot, preds, labs

# Logging
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write(f"实验：完整数据集 7:1.5:1.5 划分 | CTF + EfficientNetV2-B0\n")
    f.write("=" * 60 + "\n")
    f.write(f"数据集路径: {DATA_ROOT}\n")
    f.write(f"输出路径: {LOG_DIR}\n")
    f.write(f"训练集: {len(train_dataset)} | 验证: {len(val_dataset)} | 测试: {len(test_dataset)}\n")
    f.write("=" * 60 + "\n\n")
    f.write("epoch,train_loss,train_acc,val_loss,val_acc,residual_scale\n")

# Main loop
history = {"tl": [], "ta": [], "vl": [], "va": [], "lr": []}
best_acc = 0
start_time = time.time()

print(f"🚀 开始训练 | 完整数据集 7:1.5:1.5 | 设备: {DEVICE}")
print(f"📂 日志目录: {LOG_DIR}\n")

save_ctf_visualization(model, val_loader, VIS_SAVE_PATH, DEVICE, epoch=0)

for epoch in range(EPOCHS):
    epoch_start = time.time()
    lr = optimizer.param_groups[0]['lr']
    current_scale = model.ctf.residual_scale.item()

    tl, ta = train_one_epoch()
    vl, va, val_preds, val_labs = evaluate(val_loader, use_tta=False)
    scheduler.step()

    history["tl"].append(tl)
    history["ta"].append(ta)
    history["vl"].append(vl)
    history["va"].append(va)
    history["lr"].append(lr)

    per_class_acc = per_class_accuracy(val_labs, val_preds, NUM_CLASSES)
    per_class_str = " | ".join([f"{CLASS_NAMES[i]}: {per_class_acc[i]*100:.1f}%" for i in range(NUM_CLASSES)])
    epoch_time = time.time() - epoch_start

    gpu_info = ""
    if GPU_MONITOR:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = mem_info.used / 1024**3
        gpu_info = f" | GPU Memory: {mem_used:.2f} GB"

    print(f"Epoch {epoch + 1:03d}/{EPOCHS} | LR={lr:.6f} | CTF_Scale={current_scale:.4f} | Time: {epoch_time:.1f}s{gpu_info}")
    print(f"Train Loss: {tl:.4f} | Acc: {ta:.4f}")
    print(f"Val   Loss: {vl:.4f} | Acc: {va:.4f}")
    print(f"Val Per-Class: {per_class_str}\n")

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{epoch + 1},{tl:.4f},{ta:.4f},{vl:.4f},{va:.4f},{current_scale:.6f}\n")

    if va > best_acc:
        best_acc = va
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"✅ 最佳模型已保存！Val Acc: {best_acc:.4f}\n")
        
    if (epoch + 1) % 20 == 0:
        save_ctf_visualization(model, val_loader, VIS_SAVE_PATH, DEVICE, epoch+1)

# Evaluation
total_time = time.time() - start_time
print("=" * 60)
print(f"⏱️ 训练总耗时: {total_time/60:.1f} 分钟")

print("\n🧪 测试集评估...")
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
test_loss, test_acc, test_pred, test_true = evaluate(test_loader, use_tta=True)

report = classification_report(test_true, test_pred, target_names=CLASS_NAMES, digits=4)
print(f"✅ 测试集准确率 (TTA): {test_acc:.4f}")
print(report)

with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"测试集准确率: {test_acc:.4f}\n")
    f.write(report)

# Plotting
plt.figure(figsize=(16, 12))
plt.subplot(221)
plt.plot(history["tl"], label='Train Loss')
plt.plot(history["vl"], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.subplot(222)
plt.plot(history["ta"], label='Train Acc')
plt.plot(history["va"], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(223)
plt.plot(history["lr"], c='r')
plt.title('Learning Rate')

plt.subplot(224)
sns.heatmap(confusion_matrix(test_true, test_pred), annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Test Confusion Matrix')

plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print(f"\n🎉 全部完成！文件保存在：\n{LOG_DIR}")