import matplotlib.pyplot as plt
import numpy as np
import os


#./chirplet/logs/tf_efficientnetv2_b0+PRF/small_10/20260429_133537/NOISE_ABLATION_CTF.txt
snr = ["Clean", "20", "10", "5", "0"]
with_ctf =    [0.9586, 0.9577, 0.9065, 0.7785, 0.6675]
without_ctf = [0.9586, 0.9571, 0.9048, 0.7804, 0.6692]


save_path = "./z_comparison/p5"


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(7, 5))


plt.plot(snr, with_ctf, 'b-o', linewidth=2.5, markersize=7, label='With CTF')
plt.plot(snr, without_ctf, 'r--s', linewidth=2.5, markersize=6, label='Without CTF')


plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.ylim(0.65, 0.98)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.title('Noise Robustness Ablation of CTF Module', fontsize=13)

plt.tight_layout()


plt.savefig(os.path.join(save_path, "noise_ablation_ctf.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_path, "noise_ablation_ctf.png"), dpi=300, bbox_inches='tight')
plt.close()

print("✅✅✅ 图片已经保存成功！！")
print(f"📍 位置：{save_path}")
print("🖼️  文件名：noise_ablation_ctf.png")
print("📄  文件名：noise_ablation_ctf.pdf")