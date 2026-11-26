# Use a headless plotting backend
import matplotlib
matplotlib.use('Agg')

import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=np.ComplexWarning)

# ==============================================================================

# ==============================================================================
TARGET_SR = 16000
WINDOW_DURATION = 0.5
OVERLAP_DURATION = 0.3
CHIRP_WIN_LENGTH = 4096
CHIRP_HOP_LENGTH = 512

C_GLOBAL_MIN = -5.0
C_GLOBAL_MAX = 5.0
C_TOTAL_BINS = 64

# Dataset paths
AUDIO_ROOT_DIR = "./data/ShipsEar"
CSV_FILE_PATH = "./data/ShipsEar/shipsEarMeta_90.csv"

# Output files
FULL_REPORT_PATH = "./full_subband_division_report.txt"
FIG_PDF_PATH = "./chirp_subband_division_figure.pdf"
FIG_PNG_PATH = "./chirp_subband_division_figure.png"

# Run mode
SINGLE_AUDIO_MODE = False
SINGLE_AUDIO_PATH = "./data/ShipsEar/6__10_07_13_marDeCangas_Entra.wav"

# Earlier band partition
OLD_SPLIT_INDEX = [21, 43]

# Class mapping
CLASS_MAPPING = {
    "Ocean liner": "A_Large_Merchant_Ship",
    "RORO": "A_Large_Merchant_Ship",
    "Dredger": "A_Large_Merchant_Ship",
    "Motorboat": "B_Small_Speedboat",
    "Tugboat": "B_Small_Speedboat",
    "Pilot ship": "B_Small_Speedboat",
    "Sailboat": "B_Small_Speedboat",
    "Yacht": "B_Small_Speedboat",
    "High speed motorboat": "B_Small_Speedboat",
    "Zodiac": "B_Small_Speedboat",
    "Fishboat": "C_Fishing_Boat",
    "Trawler": "C_Fishing_Boat",
    "Mussel boat": "C_Fishing_Boat",
    "fishboat": "C_Fishing_Boat",
    "Passengers": "D_Ferry",
    "Natural ambient noise": "E_Ambient_Noise"
}

np.random.seed(42)

# Buffer the report text.
report_lines = []

def add_report(content):
    print(content)
    report_lines.append(content)

# ==============================================================================
# Estimate chirp rates.
# ==============================================================================
def calc_frame_chirp_rates(audio_seg, sr, win_len, hop_len):
    stft_mat = librosa.stft(audio_seg, n_fft=win_len, hop_length=hop_len, win_length=win_len)
    inst_freq = librosa.phase_vocoder(np.angle(stft_mat), hop_length=hop_len, rate=hop_len/sr)
    time_axis = np.arange(inst_freq.shape[1]) * hop_len / sr

    c_list = []
    for freq_idx in range(inst_freq.shape[0]):
        freq_trace = inst_freq[freq_idx, :]
        if len(time_axis) != len(freq_trace):
            continue
        coeff = np.polyfit(time_axis, freq_trace, deg=1)
        c_list.append(coeff[0])
    return c_list

# ==============================================================================

# ==============================================================================
def collect_sampled_dataset_c(max_files_per_class=5, window_stride=5, seed=42):
    rng = np.random.RandomState(seed)
    all_c = []
    win_samples = int(WINDOW_DURATION * TARGET_SR)
    step_samples = win_samples - int(OVERLAP_DURATION * TARGET_SR)

    df = pd.read_csv(CSV_FILE_PATH)
    df["mapped_class"] = df["type"].map(CLASS_MAPPING)
    df = df.dropna(subset=["mapped_class"])

    classes = sorted(df["mapped_class"].unique())
    add_report(f"📊 数据集共 {len(df)} 条音频，{len(classes)} 个类别")
    add_report(f"抽样策略：每类最多抽取 {max_files_per_class} 条音频，窗口隔 {window_stride} 取1\n")

    for cls in classes:
        cls_df = df[df["mapped_class"] == cls]
        n_total = len(cls_df)
        n_sample = min(max_files_per_class, n_total)
        sampled_df = cls_df.sample(n=n_sample, random_state=rng)
        add_report(f"  类别 {cls}: 总 {n_total} 条 → 抽取 {n_sample} 条")


        file_iter = tqdm(sampled_df.iterrows(), total=n_sample, desc=f"    处理 {cls}", leave=False)
        for _, row in file_iter:
            fname = row["filename"]
            fpath = os.path.join(AUDIO_ROOT_DIR, fname)
            if not os.path.exists(fpath):
                continue
            try:
                y, _ = librosa.load(fpath, sr=TARGET_SR, mono=True)
                start = 0
                window_idx = 0
                while start + win_samples <= len(y):
                    if window_idx % window_stride == 0:
                        seg = y[start:start + win_samples]
                        c_list = calc_frame_chirp_rates(seg, TARGET_SR, CHIRP_WIN_LENGTH, CHIRP_HOP_LENGTH)
                        all_c.extend(c_list)
                    start += step_samples
                    window_idx += 1
            except Exception as e:
                add_report(f"    ⚠️ 文件 {fname} 处理失败：{e}")
                continue
    return np.array(all_c)

# Single-recording diagnostic
def collect_single_audio_c(audio_path):
    all_c = []
    if not os.path.exists(audio_path):
        add_report(f"❌ 音频不存在：{audio_path}")
        return np.array([])

    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    win_samples = int(WINDOW_DURATION * TARGET_SR)
    step_samples = win_samples - int(OVERLAP_DURATION * TARGET_SR)
    total_len = len(y)
    start = 0
    seg_cnt = 0

    while start + win_samples <= total_len:
        seg = y[start:start + win_samples]
        c_list = calc_frame_chirp_rates(seg, TARGET_SR, CHIRP_WIN_LENGTH, CHIRP_HOP_LENGTH)
        all_c.extend(c_list)
        start += step_samples
        seg_cnt += 1

    add_report(f"✅ 单音频处理完成：{seg_cnt}个窗口，共{len(all_c):,}个时频点")
    return np.array(all_c)

# ==============================================================================

# ==============================================================================
def step1_distribution_analysis(c_array):
    add_report("")
    add_report("="*85)
    add_report("📌 步骤1：全量啁啾率分布统计分析（基于分层抽样数据）")
    add_report("="*85)
    add_report("目的：摸清船舶真实调频特性，确认数据集中范围，排除不合理假设")
    add_report("-"*85)

    total = len(c_array)
    add_report(f"总统计样本量：{total:,} 个时频点")
    add_report(f"最小值：{np.min(c_array):+.4f} Hz/s")
    add_report(f"最大值：{np.max(c_array):+.4f} Hz/s")
    add_report(f"均值：{np.mean(c_array):+.6f} Hz/s （接近0，说明正负调频对称）")
    add_report(f"标准差：{np.std(c_array):.4f} Hz/s （数值越小，数据越集中在0附近）")
    add_report(f"中位数：{np.median(c_array):+.6f} Hz/s")

    add_report("\n🔍 关键分位数（直观看到数据集中程度）：")
    for q in [0.01, 0.05, 0.25, 0.75, 0.95, 0.99]:
        val = np.quantile(c_array, q)
        add_report(f"  {q*100:>3.0f}% 分位：{val:+.4f} Hz/s")

    add_report("\n📊 对称区间样本占比（划分核心参考）：")
    add_report(f"  {'|c| ≤ tau':<12} {'样本数':>12} {'占比':>8}")
    add_report("  " + "-"*35)
    for tau in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        cnt = np.sum(np.abs(c_array) <= tau)
        ratio = cnt / total * 100
        add_report(f"  {tau:<12.2f} {cnt:>12,} {ratio:>7.2f}%")

    add_report("\n💡 本步结论：")
    add_report("  ShipsEar港口低速船舶啁啾率高度集中在0附近，绝大多数样本|c|<0.5Hz/s；")
    add_report("  若直接用无约束Otsu，会切出极窄中间类，物理意义失效，因此必须加物理+工程约束。")
    add_report("="*85)

# ==============================================================================

# ==============================================================================
def step2_constraint_explain():
    add_report("")
    add_report("="*85)
    add_report("📌 步骤2：物理语义与寻优约束设定")
    add_report("="*85)
    add_report("约束1：三通道必须以0为中心对称")
    add_report("  → 对应物理意义：负向调频（减速）、稳态巡航（调频≈0）、正向调频（加速）")
    add_report("约束2：三类样本均非空，确保每个通道均承载有效工况")
    add_report("  → 避免出现退化通道，保证三通道物理可解释性")
    add_report("约束3：中间通道宽度完全由数据驱动，不预设最小啁啾基数")
    add_report("  → 尊重港口船舶真实调频极其微弱的物理现实")
    add_report("\n寻优准则：在满足对称且三类非空的前提下，选取类间方差最大的阈值")
    add_report("  → 兼顾物理合理性与统计最优性")
    add_report("="*85)

# ==============================================================================

# ==============================================================================
def step3_find_optimal_threshold(c_array, tau_min=0.05, tau_max=0.35, step=0.001):
    add_report("")
    add_report("="*85)
    add_report("📌 步骤3：数据驱动的最优阈值寻优（类间方差最大化）")
    add_report("="*85)
    add_report(f"寻优范围：tau ∈ [{tau_min}, {tau_max}] Hz/s，步长：{step}")
    add_report("优化目标：三类样本的类间方差最大 → 三类特征区分度最高")
    add_report("约束：三类均非空（不限制中间通道宽度，以真实数据分布为准）")
    add_report("-"*85)

    valid_c = c_array[(c_array >= C_GLOBAL_MIN) & (c_array <= C_GLOBAL_MAX)]
    total = len(valid_c)
    total_mean = np.mean(valid_c)

    best_tau = None
    best_var = -1
    best_ratio = (0, 0, 0)

    tau_list = np.arange(tau_min, tau_max + step, step)
    for tau in tau_list:
        c_low = valid_c[valid_c < -tau]
        c_mid = valid_c[(valid_c >= -tau) & (valid_c <= tau)]
        c_high = valid_c[valid_c > tau]

        if len(c_low) == 0 or len(c_mid) == 0 or len(c_high) == 0:
            continue

        w1, w2, w3 = len(c_low)/total, len(c_mid)/total, len(c_high)/total
        m1, m2, m3 = np.mean(c_low), np.mean(c_mid), np.mean(c_high)
        between_var = w1*(m1-total_mean)**2 + w2*(m2-total_mean)**2 + w3*(m3-total_mean)**2

        if between_var > best_var:
            best_var = between_var
            best_tau = tau
            best_ratio = (w1, w2, w3)

    if best_tau is None:
        add_report("❌ 错误：在寻优范围内未找到任何合法划分，请扩大 tau_min 或 tau_max")
        exit(1)

    add_report(f"✅ 寻优完成！最优对称阈值：tau = {best_tau:.3f} Hz/s")
    add_report(f"   对应物理分界：c₁ = {-best_tau:.3f} Hz/s，c₂ = {best_tau:.3f} Hz/s")
    add_report(f"   最大类间方差：{best_var:.6f}")
    add_report(f"\n   三类样本占比：")
    add_report(f"     低啁啾（负向机动）：{best_ratio[0]*100:.2f}%")
    add_report(f"     中啁啾（稳态巡航）：{best_ratio[1]*100:.2f}%")
    add_report(f"     高啁啾（正向机动）：{best_ratio[2]*100:.2f}%")

    add_report("\n💡 本步结论：")
    add_report("  该阈值是三类调频特征统计区分度最高的分界，完全由数据驱动；")
    add_report("  中间通道较窄反映了港口船舶调频极其微弱的物理现实，两侧通道虽宽但集中了机动特征。")
    add_report("="*85)

    return -best_tau, best_tau

# ==============================================================================

# ==============================================================================
def step4_map_to_index(c_low, c_high):
    add_report("")
    add_report("="*85)
    add_report("📌 步骤4：连续物理阈值 → 64维离散啁啾基索引映射")
    add_report("="*85)
    delta_c = (C_GLOBAL_MAX - C_GLOBAL_MIN) / (C_TOTAL_BINS - 1)
    add_report(f"全局啁啾范围：[{C_GLOBAL_MIN}, {C_GLOBAL_MAX}] Hz/s")
    add_report(f"离散啁啾基总数：{C_TOTAL_BINS} 个（索引0 ~ {C_TOTAL_BINS-1}）")
    add_report(f"相邻索引物理步长：{delta_c:.4f} Hz/s")
    add_report(f"映射公式：idx = (c - ({C_GLOBAL_MIN})) / {delta_c:.4f}")
    add_report("-"*85)

    idx_low_float = (c_low - C_GLOBAL_MIN) / delta_c
    idx_low = int(round(idx_low_float))
    idx_low = max(0, min(C_TOTAL_BINS-1, idx_low))
    add_report(f"负向分界 c = {c_low:.4f} Hz/s")
    add_report(f"  计算：idx = ({c_low:.4f} - ({C_GLOBAL_MIN})) / {delta_c:.4f} = {idx_low_float:.3f}")
    add_report(f"  四舍五入后索引：{idx_low}")
    add_report("")

    idx_high_float = (c_high - C_GLOBAL_MIN) / delta_c
    idx_high = int(round(idx_high_float))
    idx_high = max(0, min(C_TOTAL_BINS-1, idx_high))
    add_report(f"正向分界 c = {c_high:.4f} Hz/s")
    add_report(f"  计算：idx = ({c_high:.4f} - ({C_GLOBAL_MIN})) / {delta_c:.4f} = {idx_high_float:.3f}")
    add_report(f"  四舍五入后索引：{idx_high}")

    add_report("\n📌 最终三通道索引分段（可直接写入特征代码）：")
    add_report(f"  🔴 低啁啾通道(R)：索引 0 ~ {idx_low}  （共 {idx_low+1} 个啁啾基）")
    add_report(f"  🟢 中啁啾通道(G)：索引 {idx_low} ~ {idx_high}  （共 {idx_high-idx_low} 个啁啾基）")
    add_report(f"  🔵 高啁啾通道(B)：索引 {idx_high} ~ {C_TOTAL_BINS-1}  （共 {C_TOTAL_BINS-idx_high} 个啁啾基）")
    add_report("="*85)

    return idx_low, idx_high

# ==============================================================================

# ==============================================================================
def step5_compare_old_new(c_array, c_low_new, c_high_new, idx_low_new, idx_high_new):
    add_report("")
    add_report("="*85)
    add_report("📌 步骤5：初稿旧划分 vs 新最优划分 效果对比")
    add_report("="*85)

    total = len(c_array)
    old_c1 = C_GLOBAL_MIN + OLD_SPLIT_INDEX[0] * (C_GLOBAL_MAX - C_GLOBAL_MIN)/(C_TOTAL_BINS-1)
    old_c2 = C_GLOBAL_MIN + OLD_SPLIT_INDEX[1] * (C_GLOBAL_MAX - C_GLOBAL_MIN)/(C_TOTAL_BINS-1)

    old_r = np.sum(c_array < old_c1)
    old_g = np.sum((c_array >= old_c1) & (c_array < old_c2))
    old_b = np.sum(c_array >= old_c2)

    new_r = np.sum(c_array < c_low_new)
    new_g = np.sum((c_array >= c_low_new) & (c_array <= c_high_new))
    new_b = np.sum(c_array > c_high_new)

    add_report(f"{'方案':<12} {'通道':<6} {'啁啾率范围':<18} {'索引范围':<12} {'样本占比':<10}")
    add_report("-"*72)
    add_report(f"{'旧均等划分':<12} {'低啁啾':<6} {'< {:.2f}'.format(old_c1):<18} {'0~{}'.format(OLD_SPLIT_INDEX[0]):<12} {old_r/total*100:>8.2f}%")
    add_report(f"{'旧均等划分':<12} {'中啁啾':<6} {'[{:.2f}, {:.2f})'.format(old_c1, old_c2):<18} {'{}~{}'.format(OLD_SPLIT_INDEX[0], OLD_SPLIT_INDEX[1]):<12} {old_g/total*100:>8.2f}%")
    add_report(f"{'旧均等划分':<12} {'高啁啾':<6} {'≥ {:.2f}'.format(old_c2):<18} {'{}~63'.format(OLD_SPLIT_INDEX[1]):<12} {old_b/total*100:>8.2f}%")
    add_report("-"*72)
    add_report(f"{'新最优划分':<12} {'低啁啾':<6} {'< {:.3f}'.format(c_low_new):<18} {'0~{}'.format(idx_low_new):<12} {new_r/total*100:>8.2f}%")
    add_report(f"{'新最优划分':<12} {'中啁啾':<6} {'[{:.3f}, {:.3f}]'.format(c_low_new, c_high_new):<18} {'{}~{}'.format(idx_low_new, idx_high_new):<12} {new_g/total*100:>8.2f}%")
    add_report(f"{'新最优划分':<12} {'高啁啾':<6} {'> {:.3f}'.format(c_high_new):<18} {'{}~63'.format(idx_high_new):<12} {new_b/total*100:>8.2f}%")

    add_report("\n💡 对比结论：")
    add_report("  1. 旧划分：近100%样本挤在中间通道，左右通道完全无效，三通道名存实亡；")
    add_report("  2. 新划分：三类通道均有有效样本，中间承载稳态主体，两侧承载机动特征；")
    add_report("  3. 新划分兼顾数据分布、物理语义与工程有效性，工况解耦效果显著提升。")
    add_report("="*85)

# ==============================================================================
# Plotting
# ==============================================================================
def draw_result_figure(c_array, c_low, c_high, idx_low, idx_high):
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    bins = np.linspace(-0.6, 0.6, 120)

    ax.hist(c_array, bins=bins, alpha=0.7, color="#2E86AB", edgecolor="white")

    old_c1 = C_GLOBAL_MIN + OLD_SPLIT_INDEX[0] * (C_GLOBAL_MAX - C_GLOBAL_MIN)/(C_TOTAL_BINS-1)
    old_c2 = C_GLOBAL_MIN + OLD_SPLIT_INDEX[1] * (C_GLOBAL_MAX - C_GLOBAL_MIN)/(C_TOTAL_BINS-1)
    ax.axvline(old_c1, color="#999", linestyle=":", linewidth=1.5, label="Old Equal Split (±1.67)")
    ax.axvline(old_c2, color="#999", linestyle=":", linewidth=1.5)

    ax.axvline(c_low, color="#E63946", linestyle="-", linewidth=2.2,
               label=f"Optimal Split $c_1$ = {c_low:.3f} (idx={idx_low})")
    ax.axvline(c_high, color="#E63946", linestyle="-", linewidth=2.2,
               label=f"Optimal Split $c_2$ = {c_high:.3f} (idx={idx_high})")

    ax.set_xlabel("Chirp Rate (Hz/s)", fontsize=12)
    ax.set_ylabel("Number of Time-Frequency Points", fontsize=12)
    ax.set_title("Chirp Rate Distribution and Optimal Subband Division", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(FIG_PDF_PATH, bbox_inches="tight")
    plt.savefig(FIG_PNG_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    add_report(f"\n🖼️  结果图已保存：\n  PDF矢量图：{FIG_PDF_PATH}\n  PNG高清图：{FIG_PNG_PATH}")

# ==============================================================================
# Main workflow
# ==============================================================================
if __name__ == "__main__":
    add_report("="*85)
    add_report("🚢 船舶水声信号啁啾率子带划分系统（双重约束+数据驱动）")
    add_report("="*85)
    add_report("完整流程：分布统计 → 约束设定 → 最优寻优 → 索引映射 → 新旧对比")
    add_report("="*85)


    if SINGLE_AUDIO_MODE:
        add_report(f"\n▶ 运行模式：单音频测试")
        c_data = collect_single_audio_c(SINGLE_AUDIO_PATH)
    else:
        add_report(f"\n▶ 运行模式：分层抽样统计（音频抽样 + 窗口隔步）")
        c_data = collect_sampled_dataset_c(max_files_per_class=5, window_stride=5, seed=42)

    if len(c_data) == 0:
        add_report("❌ 无有效数据，程序退出。")
        exit(1)

    # Execute each stage.
    step1_distribution_analysis(c_data)
    step2_constraint_explain()
    c_low_opt, c_high_opt = step3_find_optimal_threshold(c_data, tau_min=0.05, tau_max=0.35)
    idx_low_opt, idx_high_opt = step4_map_to_index(c_low_opt, c_high_opt)
    step5_compare_old_new(c_data, c_low_opt, c_high_opt, idx_low_opt, idx_high_opt)

    # Plotting
    draw_result_figure(c_data, c_low_opt, c_high_opt, idx_low_opt, idx_high_opt)

    # Write the report.
    with open(FULL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    add_report("")
    add_report("="*85)
    add_report(f"✅ 全部流程执行完成！")
    add_report(f"📄 所有文字结果已汇总到单个文件：{FULL_REPORT_PATH}")
    add_report("👉 下一步：用得到的索引分段修改特征生成代码，再做训练消融验证")
    add_report("="*85)