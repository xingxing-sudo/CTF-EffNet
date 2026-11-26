import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configuration
BASE_DIR = "."


TARGET_FEAT = "chirplet"
COMPARE_MODELS = [
    "tf_efficientnetv2_b0",
    "tf_efficientnetv2_b0+PRF"
]


MODEL_LABELS = {
    "tf_efficientnetv2_b0": "PTCC + EfficientNetV2-B0",
    "tf_efficientnetv2_b0+PRF": "PTCC + CTF-EffNet"
}

COLORS = {
    "tf_efficientnetv2_b0": "#d62728",
    "tf_efficientnetv2_b0+PRF": "#17becf"
}

LINE_STYLES = {
    "tf_efficientnetv2_b0": "--",
    "tf_efficientnetv2_b0+PRF": "-"
}

SUBSET = "small_10"
LOG_FILENAME = "training_log.txt"
TARGET_EPOCHS = 70

OUTPUT_DIR = os.path.join(BASE_DIR, "z_comparison", "p2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure style
mpl.use('Agg')
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.7,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 15
})

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
        target_dir = os.path.join(BASE_DIR, "reference_results", feat, "logs", "tf_efficientnetv2_b0+PRF", SUBSET)
    else:
        target_dir = os.path.join(BASE_DIR, "reference_results", feat, "logs", model, SUBSET)
    if not os.path.exists(target_dir):
        return None
    date_folders = []
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            mtime = os.path.getmtime(item_path)
            date_folders.append((mtime, item_path))
    if not date_folders:
        return None
    date_folders.sort(key=lambda x: x[0], reverse=True)
    latest_dir = date_folders[0][1]
    log_path = os.path.join(latest_dir, LOG_FILENAME)
    return log_path if os.path.exists(log_path) else None


def plot_ctf_ablation():
    print("\n🎨 绘制 CTF 模块消融实验图（标准控制变量）")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    epochs = np.arange(1, TARGET_EPOCHS+1)

    loss_data = []
    acc_data = []
    curves = {}

    for m in COMPARE_MODELS:
        log_path = get_latest_log_path(TARGET_FEAT, m)
        if not log_path:
            print(f"❌ 未找到 {m}")
            continue
        
        tl, vl, va = parse_training_log(log_path)
        vl_map = map_to_target_epochs(vl, len(vl), TARGET_EPOCHS)
        va_map = map_to_target_epochs(va, len(va), TARGET_EPOCHS)

        ax1.plot(epochs, vl_map, label=MODEL_LABELS[m], 
                 color=COLORS[m], linestyle=LINE_STYLES[m])
        ax2.plot(epochs, va_map, label=MODEL_LABELS[m], 
                 color=COLORS[m], linestyle=LINE_STYLES[m])
        
        loss_data.append(vl_map)
        acc_data.append(va_map)
        curves[m] = {"loss": vl_map, "acc": va_map}

    ax1.set_title('Validation Loss')
    ax1.set_xlabel('Epoch\n(a)')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(0, TARGET_EPOCHS)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch\n(b)')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlim(0, TARGET_EPOCHS)
    ax2.legend(loc='lower right')
    ax2.grid(True)


    axins1 = ax1.inset_axes([0.65, 0.12, 0.32, 0.35])
    for m in COMPARE_MODELS:
        axins1.plot(epochs, curves[m]["loss"], color=COLORS[m], linestyle=LINE_STYLES[m], linewidth=3)
    axins1.set_xlim(TARGET_EPOCHS-20, TARGET_EPOCHS)
    valid_loss = []
    for m in COMPARE_MODELS:
        valid_loss.extend(curves[m]["loss"][~np.isnan(curves[m]["loss"])])
    y_min_loss = min(valid_loss[-40:]) - 0.002
    y_max_loss = max(valid_loss[-40:]) + 0.002
    axins1.set_ylim(y_min_loss, y_max_loss)
    axins1.set_xticklabels([])
    axins1.set_yticklabels([])
    axins1.tick_params(axis='both', which='both', length=0)



    axins2 = ax2.inset_axes([0.65, 0.55, 0.32, 0.35])
    for m in COMPARE_MODELS:
        axins2.plot(epochs, curves[m]["acc"], color=COLORS[m], linestyle=LINE_STYLES[m], linewidth=3)
    axins2.set_xlim(TARGET_EPOCHS-20, TARGET_EPOCHS)
    valid_acc = []
    for m in COMPARE_MODELS:
        valid_acc.extend(curves[m]["acc"][~np.isnan(curves[m]["acc"])])
    y_min_acc = min(valid_acc[-40:]) - 0.002
    y_max_acc = max(valid_acc[-40:]) + 0.002
    axins2.set_ylim(y_min_acc, y_max_acc)
    axins2.set_xticklabels([])
    axins2.set_yticklabels([])
    axins2.tick_params(axis='both', which='both', length=0)


    save_path = os.path.join(OUTPUT_DIR, "Fig3_CTF_Ablation_Comparison")
    fig.savefig(save_path + ".pdf", format='pdf', bbox_inches=None)
    fig.savefig(save_path + ".png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 消融实验图已保存：{save_path}")

if __name__ == "__main__":
    plot_ctf_ablation()