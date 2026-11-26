import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import timm
from tqdm import tqdm

# ----------------------

# ----------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8

# ----------------------

# ----------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
for k in list(os.environ.keys()):
    if 'OMP_' in k or 'KMP_' in k or 'OPENBLAS_' in k or 'MKL_' in k:
        del os.environ[k]
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================

# =============================================================================
FEATURE_ORDER = ["mel", "stft", "cqt", "wst", "chirplet"]
SUBTITLES = ["(a) Mel", "(b) STFT", "(c) CQT", "(d) WST", "(e) Proposed PTCC-SPC"]
LEGEND_LABELS = ["A", "B", "C", "D", "E"]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

BASE_ROOT = "."
NUM_CLASSES = 5
MODEL_NAME = "tf_efficientnetv2_b0"
SAMPLE_PER_CLASS = 200
DEVICE = torch.device("cuda")
IMG_SIZE = 224
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# =============================================================================

# =============================================================================
def get_test_path(feat):
    if feat == "chirplet":
        return f"{BASE_ROOT}/{feat}/{feat}_picture_pro/{feat}_shipsear_small_90_test"
    else:
        return f"{BASE_ROOT}/{feat}/{feat}_picture/{feat}_shipsear_small_90_test"

def get_fixed_shared_indices():
    path = get_test_path("chirplet")
    ds = datasets.ImageFolder(path)
    indices = []
    for c in range(NUM_CLASSES):
        idx = [i for i, y in enumerate(ds.targets) if y == c]
        random.shuffle(idx)
        indices += idx[:SAMPLE_PER_CLASS]
    return indices

shared_indices = get_fixed_shared_indices()
print(f"✅ 共用样本总数：{len(shared_indices)}（每类{SAMPLE_PER_CLASS}张，测试集）")

# =============================================================================

# =============================================================================
def get_pure_backbone():
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
    
    return model

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.backbone(x)
        return x.flatten(1)

def extract_feat(model, ds, indices):
    ds_sub = Subset(ds, indices)
    loader = DataLoader(ds_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    feats, labs = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extract"):
            x = x.to(DEVICE)
            f = model(x)
            feats.append(f.cpu().numpy())
            labs.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labs)

# =============================================================================

# =============================================================================
results = []
for feat in FEATURE_ORDER:
    print(f"\n→ {feat.upper()}")
    
    test_dir = get_test_path(feat)
    ds = datasets.ImageFolder(test_dir, transform=transform)
    
    model_dir = sorted(os.listdir(f"{BASE_ROOT}/{feat}/logs/tf_efficientnetv2_b0/small_10"))[-1]
    model_path = f"{BASE_ROOT}/{feat}/logs/tf_efficientnetv2_b0/small_10/{model_dir}/best_model.pth"
    
    model = get_pure_backbone()
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    new_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model = FeatureExtractor(model).to(DEVICE)
    
    f, l = extract_feat(model, ds, shared_indices)
    
    xy = TSNE(
        n_components=2, 
        perplexity=15, 
        random_state=SEED, 
        max_iter=3000,
        learning_rate=200,
        init='pca'
    ).fit_transform(f)
    results.append((xy, l))

# =============================================================================

# =============================================================================
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.subplots_adjust(wspace=0.12, bottom=0.18)


plt.suptitle(
    "T-SNE Visualization of Different Time-Frequency Features\n(Pure EfficientNetV2-B0 Backbone | 200 Samples/Class)",
    fontsize=17, 
    y=0.98,
    fontweight="bold"
)

for i, (subtitle, (xy, l)) in enumerate(zip(SUBTITLES, results)):
    ax = axes[i]
    for c in range(5):
        mask = l == c

        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            color=COLORS[c], 
            s=14,
            alpha=1.0,
            edgecolors='none'
        )
    ax.set_title(subtitle, fontsize=14, fontweight="bold", y=-0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top','right','bottom','left']].set_visible(True)


handles = [
    plt.Line2D([], [], marker="o", color=COLORS[c], linestyle="", markersize=9)
    for c in range(5)
]
axes[-1].legend(
    handles, LEGEND_LABELS,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=14, 
    frameon=False
)

plt.tight_layout()


plt.savefig("tsne_5feat_eai_final.pdf", dpi=300, bbox_inches="tight")
plt.savefig("tsne_5feat_eai_final.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n✅ 已调整完成：")
print("1. 图例仍在最右侧")
print("2. 垂直居中，不再靠上")
print("3. 散点颜色加深、无白色边框，更清晰")
print("4. 子图标题在下、边框保留，完全符合EAAI")