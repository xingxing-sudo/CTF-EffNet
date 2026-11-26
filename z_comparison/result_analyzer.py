import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


BASE_DIR = "."


FEATURES = ["mel", "stft", "cqt", "wst", "chirplet"]


MODEL_KEYS = {
    "resnet18": "ResNet18",
    "resnet18+CTF": "ResNet18 + CTF",
    "tf_efficientnetv2_b0": "EfficientNetV2-B0",
    "tf_efficientnetv2_b0+CTF": "EfficientNetV2-B0 + CTF",
    "tf_efficientnetv2_b0+PRF": "EfficientNetV2-B0 + PRF"
}


SUBSET = "small_10"


def get_test_acc_from_folder(exp_path):

    possible_files = [
        "test_results_ttaTrue.txt",
        "test_results.txt",
        "training_log.txt"
    ]
    
    for test_file_name in possible_files:
        test_file = os.path.join(exp_path, test_file_name)
        if not os.path.exists(test_file):
            continue
            
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()


            m = re.search(r'测试集准确率.*?:\s*(0\.\d{4})', content)
            if m:
                return float(m.group(1))
                

            m = re.search(r'准确率:\s*(0\.\d{4})', content)
            if m:
                return float(m.group(1))
                

            m = re.search(r'^\s*accuracy\s+(0\.\d{4})', content, re.MULTILINE)
            if m:
                return float(m.group(1))
                

            nums = re.findall(r'0\.\d{4}', content)
            valid_nums = [float(n) for n in nums if n != '0.0000']
            if valid_nums:
                return valid_nums[-1]
                
        except Exception as e:
            continue
            
    return None


def parse_log_file(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            text = f.read()


        model_pattern = r"模型: ([^\n]*)"
        model_match = re.search(model_pattern, text)
        model_raw = model_match.group(1).strip() if model_match else None
        
        model = "Unknown"
        if model_raw:

            has_ctf = "CTF" in model_raw.upper() or "ctf" in model_raw.lower()
            has_prf = "PRF" in model_raw.upper() or "prf" in model_raw.lower()
            

            if "resnet18" in model_raw.lower() or "ResNet18" in model_raw:
                base_model = "ResNet18"
            elif "efficientnet" in model_raw.lower() or "EfficientNet" in model_raw:
                base_model = "EfficientNetV2-B0"
            else:
                base_model = model_raw
            

            if has_prf:
                model = f"{base_model} + PRF"
            elif has_ctf:
                model = f"{base_model} + CTF"
            else:
                model = base_model
        else:

            if "resnet18+CTF" in log_path:
                model = "ResNet18 + CTF"
            elif "resnet18" in log_path:
                model = "ResNet18"
            elif "efficientnetv2_b0+CTF" in log_path:
                model = "EfficientNetV2-B0 + CTF"
            elif "efficientnetv2_b0+PRF" in log_path:
                model = "EfficientNetV2-B0 + PRF"
            elif "efficientnetv2_b0" in log_path:
                model = "EfficientNetV2-B0"


        epoch_pat = r"Epochs?:\s*(\d+)"
        bs_pat = r"Batch Size:\s*(\d+)"
        lr_pat = r"学习率[:\s]*([\d\.e\-]+)"

        epochs = int(re.search(epoch_pat, text).group(1)) if re.search(epoch_pat, text) else None
        bs = int(re.search(bs_pat, text).group(1)) if re.search(bs_pat, text) else None
        lr = re.search(lr_pat, text).group(1).strip() if re.search(lr_pat, text) else None

        return {
            "model": model,
            "epochs": epochs,
            "batch_size": bs,
            "lr": lr
        }
    except Exception as e:
        print(f"⚠️ 解析日志失败 {log_path}: {e}")
        return None


def collect_all_results():
    results = []

    for feat in FEATURES:

        for model_key, model_name in MODEL_KEYS.items():
            log_dir = os.path.join(BASE_DIR, "reference_results", feat, "logs", model_key, SUBSET)
            if not os.path.exists(log_dir):
                continue


            for exp_folder in os.listdir(log_dir):
                exp_path = os.path.join(log_dir, exp_folder)
                log_file = os.path.join(exp_path, "training_log.txt")
                if os.path.exists(log_file):
                    print(f"✅ 解析: {feat:10s} | {model_name:25s} | {exp_folder}")
                    info = parse_log_file(log_file)
                    if info:
                        test_acc = get_test_acc_from_folder(exp_path)
                        results.append({
                            "feature": feat.upper(),
                            "model": info["model"],
                            "test_acc": test_acc,
                            "epochs": info["epochs"],
                            "batch_size": info["batch_size"],
                            "lr": info["lr"],
                            "source_file": log_file
                        })

    df = pd.DataFrame(results)
    
    if len(df) > 0 and all(c in df.columns for c in ["model", "test_acc"]):
        df = df.sort_values(by=["model", "test_acc"], ascending=[True, False], na_position="last")
        
    return df


def generate_paper_tables(df):
    output_dir = os.path.join(BASE_DIR, "z_comparison/results")
    os.makedirs(output_dir, exist_ok=True)


    df_full = df[["feature", "model", "test_acc", "epochs", "batch_size", "lr", "source_file"]].copy()
    df_full.to_csv(os.path.join(output_dir, "ALL_EXPERIMENTS_FULL_INFO.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(output_dir, "ALL_EXPERIMENTS_FULL_INFO.txt"), 'w', encoding='utf-8') as f:
        f.write(df_full.to_string(index=False))


    resnet_models = ["ResNet18", "ResNet18 + CTF"]
    df_resnet = df[df["model"].isin(resnet_models)].copy()
    

    resnet_pivot = df_resnet.pivot_table(index='feature', columns='model', values='test_acc', aggfunc='mean')
    resnet_pivot = resnet_pivot.sort_values("ResNet18 + CTF", ascending=False) if "ResNet18 + CTF" in resnet_pivot.columns else resnet_pivot.sort_values("ResNet18", ascending=False)
    

    if "ResNet18" in resnet_pivot.columns and "ResNet18 + CTF" in resnet_pivot.columns:
        resnet_pivot["CTF提升"] = resnet_pivot["ResNet18 + CTF"] - resnet_pivot["ResNet18"]


    eff_models = ["EfficientNetV2-B0", "EfficientNetV2-B0 + CTF", "EfficientNetV2-B0 + PRF"]
    df_eff = df[df["model"].isin(eff_models)].copy()
    

    eff_pivot = df_eff.pivot_table(index='feature', columns='model', values='test_acc', aggfunc='mean')
    if "EfficientNetV2-B0 + CTF" in eff_pivot.columns:
        eff_pivot = eff_pivot.sort_values("EfficientNetV2-B0 + CTF", ascending=False)
    elif "EfficientNetV2-B0 + PRF" in eff_pivot.columns:
        eff_pivot = eff_pivot.sort_values("EfficientNetV2-B0 + PRF", ascending=False)
    else:
        eff_pivot = eff_pivot.sort_values("EfficientNetV2-B0", ascending=False)
    

    if "EfficientNetV2-B0" in eff_pivot.columns:
        if "EfficientNetV2-B0 + CTF" in eff_pivot.columns:
            eff_pivot["CTF提升"] = eff_pivot["EfficientNetV2-B0 + CTF"] - eff_pivot["EfficientNetV2-B0"]
        if "EfficientNetV2-B0 + PRF" in eff_pivot.columns:
            eff_pivot["PRF提升"] = eff_pivot["EfficientNetV2-B0 + PRF"] - eff_pivot["EfficientNetV2-B0"]


    all_models = ["ResNet18", "ResNet18 + CTF", "EfficientNetV2-B0", "EfficientNetV2-B0 + CTF", "EfficientNetV2-B0 + PRF"]
    cross = []
    for feat in df["feature"].unique():
        sub = df[df["feature"] == feat]
        row = {"Feature": feat}
        
        for m in all_models:
            vals = sub[sub["model"] == m]["test_acc"].dropna().values
            row[m] = round(np.mean(vals), 4) if len(vals) > 0 else None
            

        if row.get("ResNet18") and row.get("ResNet18 + CTF"):
            row["R18→R18+CTF"] = round(row["ResNet18 + CTF"] - row["ResNet18"], 4)
        if row.get("EfficientNetV2-B0") and row.get("EfficientNetV2-B0 + CTF"):
            row["Eff→Eff+CTF"] = round(row["EfficientNetV2-B0 + CTF"] - row["EfficientNetV2-B0"], 4)
        if row.get("EfficientNetV2-B0") and row.get("EfficientNetV2-B0 + PRF"):
            row["Eff→Eff+PRF"] = round(row["EfficientNetV2-B0 + PRF"] - row["EfficientNetV2-B0"], 4)
            
        cross.append(row)
        
    df_cross = pd.DataFrame(cross)
    

    sort_col = "EfficientNetV2-B0 + CTF" if "EfficientNetV2-B0 + CTF" in df_cross.columns else "EfficientNetV2-B0"
    if sort_col in df_cross.columns:
        df_cross = df_cross.sort_values(sort_col, ascending=False)


    tables = {
        "Table1_ResNet18_系列对比": resnet_pivot,
        "Table2_EfficientNet_系列对比": eff_pivot,
        "Table3_全模型大对比": df_cross
    }

    for name, table in tables.items():
        txt_path = os.path.join(output_dir, f"{name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(table.to_string())

        csv_path = os.path.join(output_dir, f"{name}.csv")
        table.to_csv(csv_path, encoding='utf-8-sig')

    print(f"\n📊 表格已保存至: {output_dir}")
    return tables, output_dir


def plot_comparison(df, save_dir):
    try:
        plt.rcParams['font.size'] = 12
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        valid_features = df["feature"].unique()
        

        colors = {
            "ResNet18": "#1f77b4",
            "ResNet18 + CTF": "#ff7f0e",
            "EfficientNetV2-B0": "#2ca02c",
            "EfficientNetV2-B0 + CTF": "#d62728",
            "EfficientNetV2-B0 + PRF": "#9467bd"
        }
        

        ax1 = axes[0, 0]
        all_models = ["ResNet18", "ResNet18 + CTF", "EfficientNetV2-B0", "EfficientNetV2-B0 + CTF", "EfficientNetV2-B0 + PRF"]
        
        x = np.arange(len(valid_features))
        width = 0.15
        
        for i, model in enumerate(all_models):
            accs = []
            for f in valid_features:
                vals = df[(df["model"] == model) & (df["feature"] == f)]["test_acc"].dropna().values
                accs.append(np.mean(vals) * 100 if len(vals) > 0 else np.nan)
            
            if not all(np.isnan(accs)):
                ax1.bar(x + (i - 2) * width, accs, width, label=model, color=colors.get(model, None))
        
        ax1.set_title("Full Model Comparison (All 5 Models)", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_features, rotation=10)
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        ax1.set_ylim([70, 100])


        ax2 = axes[0, 1]
        r18_acc = []
        r18_ctf_acc = []
        plot_feats = []
        
        for f in valid_features:
            r18 = df[(df["model"] == "ResNet18") & (df["feature"] == f)]["test_acc"].dropna().values
            r18_ctf = df[(df["model"] == "ResNet18 + CTF") & (df["feature"] == f)]["test_acc"].dropna().values
            
            if len(r18) > 0 and len(r18_ctf) > 0:
                r18_acc.append(np.mean(r18) * 100)
                r18_ctf_acc.append(np.mean(r18_ctf) * 100)
                plot_feats.append(f)
        
        x2 = np.arange(len(plot_feats))
        width2 = 0.35
        ax2.bar(x2 - width2/2, r18_acc, width2, label='ResNet18', color=colors["ResNet18"])
        ax2.bar(x2 + width2/2, r18_ctf_acc, width2, label='ResNet18 + CTF', color=colors["ResNet18 + CTF"])
        ax2.set_title("ResNet18 vs ResNet18 + CTF", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(plot_feats, rotation=10)
        ax2.legend()
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        ax2.set_ylim([70, 100])


        ax3 = axes[1, 0]
        eff_acc = []
        eff_ctf_acc = []
        eff_prf_acc = []
        plot_feats_eff = []
        
        for f in valid_features:
            eff = df[(df["model"] == "EfficientNetV2-B0") & (df["feature"] == f)]["test_acc"].dropna().values
            eff_ctf = df[(df["model"] == "EfficientNetV2-B0 + CTF") & (df["feature"] == f)]["test_acc"].dropna().values
            eff_prf = df[(df["model"] == "EfficientNetV2-B0 + PRF") & (df["feature"] == f)]["test_acc"].dropna().values
            
            if len(eff) > 0:
                eff_acc.append(np.mean(eff) * 100)
                eff_ctf_acc.append(np.mean(eff_ctf) * 100 if len(eff_ctf) > 0 else np.nan)
                eff_prf_acc.append(np.mean(eff_prf) * 100 if len(eff_prf) > 0 else np.nan)
                plot_feats_eff.append(f)
        
        x3 = np.arange(len(plot_feats_eff))
        width3 = 0.25
        ax3.bar(x3 - width3, eff_acc, width3, label='EfficientNetV2-B0', color=colors["EfficientNetV2-B0"])
        if not all(np.isnan(eff_ctf_acc)):
            ax3.bar(x3, eff_ctf_acc, width3, label='EfficientNetV2-B0 + CTF', color=colors["EfficientNetV2-B0 + CTF"])
        if not all(np.isnan(eff_prf_acc)):
            ax3.bar(x3 + width3, eff_prf_acc, width3, label='EfficientNetV2-B0 + PRF', color=colors["EfficientNetV2-B0 + PRF"])
        
        ax3.set_title("EfficientNetV2 Series Comparison", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_xticks(x3)
        ax3.set_xticklabels(plot_feats_eff, rotation=10)
        ax3.legend()
        ax3.grid(axis='y', linestyle='--', alpha=0.4)
        ax3.set_ylim([70, 100])


        ax4 = axes[1, 1]
        improvements = []
        feat_labels = []
        
        for f in valid_features:
            # ResNet18 → ResNet18+CTF
            r18 = df[(df["model"] == "ResNet18") & (df["feature"] == f)]["test_acc"].mean()
            r18_ctf = df[(df["model"] == "ResNet18 + CTF") & (df["feature"] == f)]["test_acc"].mean()
            
            # Eff → Eff+CTF
            eff = df[(df["model"] == "EfficientNetV2-B0") & (df["feature"] == f)]["test_acc"].mean()
            eff_ctf = df[(df["model"] == "EfficientNetV2-B0 + CTF") & (df["feature"] == f)]["test_acc"].mean()
            
            # Eff → Eff+PRF
            eff_prf = df[(df["model"] == "EfficientNetV2-B0 + PRF") & (df["feature"] == f)]["test_acc"].mean()
            
            if not np.isnan(r18) and not np.isnan(r18_ctf):
                improvements.append((r18_ctf - r18) * 100)
                feat_labels.append(f"{f}\n(R18→R18+CTF)")
            
            if not np.isnan(eff) and not np.isnan(eff_ctf):
                improvements.append((eff_ctf - eff) * 100)
                feat_labels.append(f"{f}\n(Eff→Eff+CTF)")
            
            if not np.isnan(eff) and not np.isnan(eff_prf):
                improvements.append((eff_prf - eff) * 100)
                feat_labels.append(f"{f}\n(Eff→Eff+PRF)")
        
        if improvements:
            colors_imp = ['#ff7f0e' if 'R18' in label else '#d62728' if 'Eff+CTF' in label else '#9467bd' for label in feat_labels]
            bars = ax4.bar(feat_labels, improvements, color=colors_imp)
            ax4.set_title("Module Improvement (percentage points)", fontsize=14, fontweight='bold')
            ax4.set_ylabel("Improvement (pp)")
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax4.tick_params(axis='x', rotation=45, labelsize=9)
            ax4.grid(axis='y', linestyle='--', alpha=0.4)
            

            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "comparison_plot_with_CTF_PRF.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 增强版对比图已保存: {plot_path}")
    except Exception as e:
        print(f"⚠️ 绘图跳过: {e}")
        import traceback
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    print("=" * 70)
    print(" 📊 论文实验结果自动分析器（增强版 - 支持CTF/PRF）")
    print("=" * 70)

    df = collect_all_results()
    if len(df) == 0:
        print("❌ 未找到任何日志！请检查路径是否正确。")
        exit()

    print("\n📌 所有实验汇总：")
    print(df[["feature", "model", "test_acc", "epochs", "batch_size", "lr"]].to_string(index=False))

    tables, save_dir = generate_paper_tables(df)
    plot_comparison(df, save_dir)

    print("\n🎉 分析完成！")