import numpy as np
import matplotlib.pyplot as plt

# =======================
# 1. 生成样例粒子分布
# =======================
np.random.seed(42)
Lx = 0.034  # 仿真域长度 (m)
Ly = 0.01   # 高度 (m)
N = 300     # 粒子数量

# 初始均匀分布
x0 = np.random.uniform(0, Lx, N)
y0 = np.random.uniform(0, Ly, N)

# 模拟声场驱动后，粒子向压力节点聚集
x1 = x0 + 0.001 * np.sin(2 * np.pi * x0 / (Lx / 2))  # 简单位移模型
x1 = np.clip(x1, 0, Lx)  # 限制在域内

# =======================
# 2. 统计局部数密度
# =======================
num_bins = 50
bins = np.linspace(0, Lx, num_bins + 1)

hist0, _ = np.histogram(x0, bins=bins)
hist1, _ = np.histogram(x1, bins=bins)

n0 = np.mean(hist0)
rel_change = (hist1 - n0) / n0 * 100  # 相对变化百分比
bin_centers = (bins[:-1] + bins[1:]) / 2

# =======================
# 3. 平滑处理 (双边滤波：保峰又顺滑)
# =======================
# 双边滤波结合了空间距离和数值差异的权重，对峰值与边缘有更强的保护，避免削峰
sigma_spatial_bins = 0.1          # 空间标准差（单位：bin）
sigma_intensity = max(1e-6, np.std(rel_change) * 0.5)  # 强度标准差（单位：百分比）
radius = int(3 * sigma_spatial_bins)
spatial_x = np.arange(-radius, radius + 1)
spatial_kernel = np.exp(-0.5 * (spatial_x / sigma_spatial_bins) ** 2)

# 反射填充，降低边界伪影
padded = np.pad(rel_change, pad_width=radius, mode='reflect')
rel_change_smooth = np.empty_like(rel_change)

for i in range(rel_change.size):
    center_val = padded[i + radius]
    window = padded[i:i + 2 * radius + 1]
    intensity_weights = np.exp(-0.5 * ((window - center_val) / sigma_intensity) ** 2)
    weights = spatial_kernel * intensity_weights
    denom = weights.sum()
    if denom <= 0:
        rel_change_smooth[i] = center_val
    else:
        rel_change_smooth[i] = (weights * window).sum() / denom

# =======================
# 4. 可视化结果
# =======================
plt.figure(figsize=(9, 4))

# 左图：粒子分布
plt.subplot(1, 2, 1)
plt.scatter(x1, y0, s=5, color='black')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Particle distribution after displacement")

# 右图：浓度分布曲线（含平滑处理）
plt.subplot(1, 2, 2)
plt.plot(bin_centers * 1000, rel_change, 'b:', alpha=0.4, label="Raw data")
plt.plot(bin_centers * 1000, rel_change_smooth, 'r-', linewidth=2, label="Smoothed")
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("x (mm)")
plt.ylabel("Relative change of concentration (%)")
plt.title("Concentration profile")
plt.legend()

plt.tight_layout()
plt.show()
