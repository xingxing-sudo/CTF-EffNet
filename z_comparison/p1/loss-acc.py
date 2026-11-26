import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration
BASE_DIR = "."

FEATURES = ["mel", "stft", "cqt", "wst", "chirplet"]
FEATURE_LABELS = {
    "mel": "Mel",
    "stft": "STFT",
    "cqt": "CQT",
    "wst": "WST",
    "chirplet": "PTCC-Spec (Ours)"
}

LINE_STYLES = {
    "mel": "-",
    "stft": "--",
    "cqt": "-.",
    "wst": ":",
    "chirplet": "-",
    "resnet18": "-",
    "tf_efficientnetv2_b0": "--",
    "tf_efficientnetv2_b0+PRF": "-."
}

MODELS = ["resnet18", "tf_efficientnetv2_b0", "tf_efficientnetv2_b0+PRF"]


MODEL_LABELS = {
    "resnet18": "PTCC-Spec + ResNet18",
    "tf_efficientnetv2_b0": "PTCC-Spec + EfficientNetV2-B0",
    "tf_efficientnetv2_b0+PRF": "PTCC-Spec + CTF-EffNet"
}

SUBSET = "small_10"
LOG_FILENAME = "training_log.txt"
TARGET_EPOCHS = 100

OUTPUT_DIR = os.path.join(BASE_DIR, "z_comparison", "p1/100")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure style
mpl.use('Agg')
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.7,
    "grid.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 15
})

COLORS = {
    "mel": "#1f77b4",
    "stft": "#ff7f0e",
    "cqt": "#2ca02c",
    "wst": "#9467bd",
    "chirplet": "#d62728",
    "resnet18": "#1f77b4",
    "tf_efficientnetv2_b0": "#d62728",
    "tf_efficientnetv2_b0+PRF": "#17becf"
}

# Log parser
def parse_training_log(log_path):
    train_loss = []
    val_loss = []
    val_acc = []

    if not os.path.exists(log_path):
        return np.array(train_loss), np.array(val_loss), np.array(val_acc)

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header_found = False
            for row in reader:
                if not row or len(row) < 5:
                    continue
                if row[0].strip() == "epoch" and "train_loss" in row:
                    header_found = True
                    continue
                if header_found:
                    try:
                        tl = float(row[1])
                        vl = float(row[3])
                        va = float(row[4])
                        train_loss.append(tl)
                        val_loss.append(vl)
                        val_acc.append(va)
                    except:
                        continue
        return np.array(train_loss), np.array(val_loss), np.array(val_acc)
    except:
        return np.array(train_loss), np.array(val_loss), np.array(val_acc)


def map_to_target_epochs(original_series, original_len, target_len):
    new_series = np.full(target_len, np.nan)
    for i in range(target_len):
        orig_idx = int(round(i / target_len * original_len))
        orig_idx = min(orig_idx, original_len - 1)
        new_series[i] = original_series[orig_idx]
    return new_series


def get_latest_log_path(feat, model):
    if model == "tf_efficientnetv2_b0+PRF":
        small_10_dir = os.path.join(BASE_DIR, "reference_results", feat, "logs", "tf_efficientnetv2_b0+PRF", SUBSET)
    else:
        small_10_dir = os.path.join(BASE_DIR, "reference_results", feat, "logs", model, SUBSET)

    if not os.path.exists(small_10_dir):
        return None

    date_folders = []
    for item in os.listdir(small_10_dir):
        item_path = os.path.join(small_10_dir, item)
        if os.path.isdir(item_path):
            mtime = os.path.getmtime(item_path)
            date_folders.append((mtime, item_path))

    if not date_folders:
        return None

    date_folders.sort(key=lambda x: x[0], reverse=True)
    latest_dir = date_folders[0][1]
    log_path = os.path.join(latest_dir, LOG_FILENAME)
    return log_path if os.path.exists(log_path) else None


def collect_all_experiments():
    data = {}
    print("=" * 80)
    print("🔍 扫描所有实验...")
    print("=" * 80)

    for feat in FEATURES:
        for model in MODELS:
            print(f"\n📍 处理: {feat.upper():10s} | {MODEL_LABELS[model]:25s}")
            log_path = get_latest_log_path(feat, model)

            if log_path is None:
                print(f"   ❌ 未找到日志文件")
                continue

            print(f"   ✅ 找到日志: {os.path.basename(os.path.dirname(log_path))}")
            tl, vl, va = parse_training_log(log_path)

            if len(tl) == 0:
                print(f"   ❌ 无有效数据")
                continue

            tl_map = map_to_target_epochs(tl, len(tl), TARGET_EPOCHS)
            vl_map = map_to_target_epochs(vl, len(vl), TARGET_EPOCHS)
            va_map = map_to_target_epochs(va, len(va), TARGET_EPOCHS)

            data[(feat, model)] = {
                "train_loss": tl_map,
                "val_loss": vl_map,
                "val_acc": va_map,
                "original_epochs": len(tl)
            }
            print(f"   ✅ 映射完成：原{len(tl)}轮 → 70轮")

    print(f"\n✅ 总共收集到 {len(data)} 组有效实验数据")
    return data


def set_adaptive_ylim(ax, data_list):
    all_vals = []
    for d in data_list:
        all_vals.extend(d[~np.isnan(d)])
    if not all_vals:
        return
    min_val = min(all_vals)
    max_val = max(all_vals)
    rng = max_val - min_val
    ax.set_ylim(min_val - rng*0.08, max_val + rng*0.05)


def save_figure(fig, base_name):
    pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
    fig.savefig(pdf_path, format='pdf')
    fig.savefig(png_path, dpi=300)
    plt.close()
    print(f"   ✅ 已保存期刊标准图：{base_name}")


def plot_effnet_feature_comparison(data):
    print("\n🎨 绘制图1: EfficientNetV2-B0 特征对比")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = np.arange(1, TARGET_EPOCHS+1)
    target_model = "tf_efficientnetv2_b0"

    loss_data = []
    acc_data = []

    for feat in FEATURES:
        key = (feat, target_model)
        if key not in data: continue
        d = data[key]
        ax1.plot(epochs, d['train_loss'], label=FEATURE_LABELS[feat],
                 color=COLORS[feat], linestyle=LINE_STYLES[feat])
        ax2.plot(epochs, d['val_acc'], label=FEATURE_LABELS[feat],
                 color=COLORS[feat], linestyle=LINE_STYLES[feat])
        loss_data.append(d['train_loss'])
        acc_data.append(d['val_acc'])

    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch\n(a)')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(0, TARGET_EPOCHS)
    set_adaptive_ylim(ax1, loss_data)
    ax1.legend(loc='upper right')

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch\n(b)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlim(0, TARGET_EPOCHS)
    set_adaptive_ylim(ax2, acc_data)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    save_figure(fig, "Fig1_EffNet_Feature_Comparison")


def plot_chirplet_model_comparison(data):
    print("\n🎨 绘制图2: PTCC 模型对比（含CTF）")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs = np.arange(1, TARGET_EPOCHS+1)
    target_feat = "chirplet"

    loss_data = []
    acc_data = []

    for model in MODELS:
        key = (target_feat, model)
        if key not in data: continue
        d = data[key]
        ax1.plot(epochs, d['train_loss'], label=MODEL_LABELS[model],
                 color=COLORS[model], linestyle=LINE_STYLES[model])
        ax2.plot(epochs, d['val_acc'], label=MODEL_LABELS[model],
                 color=COLORS[model], linestyle=LINE_STYLES[model])
        loss_data.append(d['train_loss'])
        acc_data.append(d['val_acc'])

    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch\n(a)')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(0, TARGET_EPOCHS)
    set_adaptive_ylim(ax1, loss_data)
    ax1.legend(loc='upper right')

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch\n(b)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlim(0, TARGET_EPOCHS)
    set_adaptive_ylim(ax2, acc_data)
    ax2.legend(loc='lower right')

    plt.tight_layout()
    save_figure(fig, "Fig2_PTCCSpec_Model_Comparison_CTF")


if __name__ == "__main__":
    data = collect_all_experiments()
    if data:
        plot_effnet_feature_comparison(data)
        plot_chirplet_model_comparison(data)
        print("\n" + "=" * 80)
        print("🎉 绘图完成：期刊标准300DPI + 适中画布 + 不爆内存")
        print("=" * 80)
    else:
        print("\n❌ 无有效数据")