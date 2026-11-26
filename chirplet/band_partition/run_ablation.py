#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

# Global configuration

FEATURE_BASE = os.path.abspath("./chirplet/band_partition/chirplet_features")

# Output directory for the dataset splits
SPLIT_BASE = "./chirplet/band_partition/ablation_splits"

# Output directory for training logs and checkpoints
ABLATION_LOG_ROOT = "./chirplet/band_partition/ablation_logs"

# Results file
RESULT_CSV = os.path.join(ABLATION_LOG_ROOT, "ablation_results.csv")

# Mapping from scheme names to directories
SCHEME_DIRS = {
    "band_1bin":  os.path.join(FEATURE_BASE, "band_1bin"),
    "band_3bin":  os.path.join(FEATURE_BASE, "band_3bin"),
    "band_5bin":  os.path.join(FEATURE_BASE, "band_5bin"),
    "band_22bin": os.path.join(FEATURE_BASE, "band_22bin"),
    "band_nosplit": os.path.join(FEATURE_BASE, "band_nosplit"),
}

# Class names
CLASS_NAMES = [
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise"
]
NUM_CLASSES = 5

# Dataset split ratios
TRAIN_RATIO = 0.07
VAL_RATIO   = 0.015
TEST_RATIO  = 0.915

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 300
LR = 1e-4
IMG_SIZE = 224
MODEL_NAME = "tf_efficientnetv2_b0"
RANDOM_SEED = 456

# MixUp
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(RANDOM_SEED)

# Dataset splitting helpers
def get_all_image_paths(feature_dir):
    """Return the feature image paths and their class labels."""
    paths = []
    labels = []
    class_dirs = sorted(os.listdir(feature_dir))
    for cls_name in class_dirs:
        cls_dir = os.path.join(feature_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(CLASS_NAMES.index(cls_name))
    return paths, labels

def stratified_split(paths, labels, train_r, val_r, test_r, seed):
    """Return stratified training, validation and test index lists."""
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, (p, l) in enumerate(zip(paths, labels)):
        class_indices[l].append(idx)
    
    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.RandomState(seed)
    for cls, indices in class_indices.items():
        n = len(indices)
        n_train = max(1, int(np.round(n * train_r)))
        n_val   = max(1, int(np.round(n * val_r)))
        n_test  = n - n_train - n_val
        shuffled = rng.permutation(indices)
        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train+n_val])
        test_idx.extend(shuffled[n_train+n_val:])
    return train_idx, val_idx, test_idx

def create_split_dataset(feature_dir, output_dir, split_info=None, save_split_info=False, seed=42):
    """Create a new partition when no split is supplied, or materialize the existing split."""
    # Create the output directory if needed.
    os.makedirs(output_dir, exist_ok=True)
    
    paths, labels = get_all_image_paths(feature_dir)
    if split_info is None:
        train_idx, val_idx, test_idx = stratified_split(paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed)

        rel_paths = [os.path.relpath(p, feature_dir) for p in paths]
        split_info = {
            'train': [rel_paths[i] for i in train_idx],
            'val':   [rel_paths[i] for i in val_idx],
            'test':  [rel_paths[i] for i in test_idx],
        }
        if save_split_info:
            with open(os.path.join(output_dir, "split_info.json"), 'w') as f:
                json.dump(split_info, f)
    
    # Create the split directory structure.
    for split in ['train', 'val', 'test']:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    

    for split in ['train', 'val', 'test']:
        for rel_path in split_info[split]:
            src = os.path.join(feature_dir, rel_path)
            dst = os.path.join(output_dir, split, rel_path)
            if not os.path.exists(dst):
                os.symlink(src, dst)


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
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

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
        chirp_feat = nn.functional.conv2d(x, kernels, padding=self.kernel_size//2, groups=C)
        chirp_feat = self.bn_after_conv(chirp_feat)
        residual = self.fusion(chirp_feat)
        return x + torch.clamp(self.residual_scale, min=0.0) * residual

class ModelWithCTF(nn.Module):
    def __init__(self, ctf, backbone):
        super().__init__()
        self.ctf = ctf
        self.backbone = backbone
    def forward(self, x):
        x = self.ctf(x)
        return self.backbone(x)

# Data augmentation
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

# Model training
def train_scheme(scheme_name, data_root, log_root):
    # Create the log directory.
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_root, scheme_name, time_str)
    os.makedirs(log_dir, exist_ok=True)
    model_save_path = os.path.join(log_dir, "best_model.pth")

    # Data loading
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

    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform_train)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform_val)
    test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), transform_val)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE*2, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE*2, shuffle=False, num_workers=8, pin_memory=True)


    set_seed(RANDOM_SEED)
    backbone = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, in_chans=3,
        drop_rate=0.2, drop_path_rate=0.1,
    )
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(backbone.num_features, NUM_CLASSES)
    )
    ctf = CTFModule(in_channels=3, n_kernels=32, kernel_size=15)
    model = ModelWithCTF(ctf, backbone).to(DEVICE)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_epochs = 10
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_epochs])

    # Mixup
    from timm.data.mixup import Mixup
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=MIXUP_PROB,
        switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=NUM_CLASSES
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"{scheme_name} Epoch {epoch+1}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
            if len(y.shape) == 2:
                pred_label = pred.argmax(1)
                true_label = y.argmax(1)
            else:
                pred_label = pred.argmax(1)
                true_label = y
            train_correct += (pred_label == true_label).sum().item()
            train_total += x.size(0)


        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (pred.argmax(1) == y).sum().item()
                val_total += x.size(0)

        scheduler.step()
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)

        print(f"[{scheme_name}] Epoch {epoch+1:03d} | Train Loss: {train_loss/train_total:.4f} Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")

    # Evaluate the best checkpoint.
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Final Test", leave=False):
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu()
            all_preds.extend(pred.numpy())
            all_labels.extend(y.numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    macro_f1 = report['macro avg']['f1-score']

    # Write the experiment log.
    with open(os.path.join(log_dir, "result.txt"), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    return acc, macro_f1

# Main workflow
def main():
    os.makedirs(SPLIT_BASE, exist_ok=True)
    os.makedirs(ABLATION_LOG_ROOT, exist_ok=True)


    ref_scheme = "band_1bin"
    ref_dir = SCHEME_DIRS[ref_scheme]
    ref_split_dir = os.path.join(SPLIT_BASE, ref_scheme)
    if not os.path.exists(os.path.join(ref_split_dir, "split_info.json")):
        print(f"📂 生成统一划分信息，基于 {ref_scheme} ...")
        create_split_dataset(ref_dir, ref_split_dir, split_info=None, save_split_info=True, seed=RANDOM_SEED)
    else:
        print(f"📂 划分信息已存在，跳过生成。")


    with open(os.path.join(ref_split_dir, "split_info.json"), 'r') as f:
        split_info = json.load(f)


    for scheme_name, src_dir in SCHEME_DIRS.items():
        if scheme_name == ref_scheme:
            continue
        dst_split_dir = os.path.join(SPLIT_BASE, scheme_name)
        if not os.path.exists(os.path.join(dst_split_dir, "split_info.json")):
            print(f"📂 为 {scheme_name} 创建划分...")
            create_split_dataset(src_dir, dst_split_dir, split_info=split_info, seed=RANDOM_SEED)
        else:
            print(f"📂 {scheme_name} 划分已存在，跳过。")


    results = []
    for scheme_name in SCHEME_DIRS:
        data_root = os.path.join(SPLIT_BASE, scheme_name)
        print(f"\n{'='*60}\n🚀 开始训练方案：{scheme_name}\n{'='*60}")
        acc, macro_f1 = train_scheme(scheme_name, data_root, ABLATION_LOG_ROOT)
        results.append({
            "scheme": scheme_name,
            "accuracy": acc,
            "macro_f1": macro_f1
        })
        print(f"✅ {scheme_name} 完成 | Test Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")


    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["scheme", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📊 所有消融实验完成，结果已保存至 {RESULT_CSV}")
    for r in results:
        print(f"{r['scheme']:15s}  Acc: {r['accuracy']:.4f}  Macro-F1: {r['macro_f1']:.4f}")

if __name__ == "__main__":
    main()