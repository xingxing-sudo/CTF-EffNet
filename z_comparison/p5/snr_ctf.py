import matplotlib.pyplot as plt
import numpy as np
import os



snr = ["0", "5", "10", "20", "Clean"]

with_ctf = [0.1920, 0.2080, 0.5780, 0.9540, 0.9640]


save_path = "./z_comparison/p5"


plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.figure(figsize=(7, 5))


plt.plot(snr, with_ctf, 'b-o', linewidth=3, markersize=8, label='Proposed CTF-EffNet')


plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0.15, 1.0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='best')
plt.title('Noise Robustness of the Proposed Model', fontsize=13)

plt.tight_layout()


plt.savefig(os.path.join(save_path, "snr_ctf.pdf"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_path, "snr_ctf.png"), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 横轴SNR升序排列，噪声鲁棒性图绘制完成！")
print(f"📍 保存路径：{save_path}")