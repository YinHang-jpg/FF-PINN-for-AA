import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

# 导入ARF常量
try:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
except Exception:
    from ARF import p_0, rho_0, gamma, c_0, frequency, amplitude


class ARFNetX(nn.Module):
    """空间因子网络：输入 [x]，输出 [空间因子]"""
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算波数用于傅里叶特征
        wavelength = c_0 / frequency  # 0.034 m
        k = 2 * np.pi / wavelength    # 约 184.8 rad/m
        
        # 傅里叶特征权重（多个频率分量）
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
        
        # 主网络
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2, 256),  # sin和cos特征
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 生成傅里叶特征
        fourier_x = self.fourier_weights * x
        fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        
        # 通过主网络
        return self.layers(fourier_features)


class ARFNetT(nn.Module):
    """时间因子网络：输入 [t]，输出 [时间因子]"""
    def __init__(self, fourier_features=16):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算角频率用于傅里叶特征
        omega = 2 * np.pi * frequency  # 约 2.51e6 rad/s
        
        # 傅里叶特征权重（多个频率分量）
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * omega)
        
        # 主网络
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2, 128),  # sin和cos特征
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, t):
        # 生成傅里叶特征
        fourier_t = self.fourier_weights * t
        fourier_features = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        
        # 通过主网络
        return self.layers(fourier_features)


def load_models_and_params():
    """加载训练好的空间和时间模型及其归一化参数"""
    # 加载空间模型
    model_x = ARFNetX(fourier_features=32)
    model_x.load_state_dict(torch.load('arf_model_x.pth', map_location='cpu'))
    model_x.eval()
    
    # 加载时间模型
    model_t = ARFNetT(fourier_features=16)
    model_t.load_state_dict(torch.load('arf_model_t.pth', map_location='cpu'))
    model_t.eval()
    
    # 加载空间模型归一化参数
    with open('arf_model_x_normalization_params.json', 'r') as f:
        norm_params_x = json.load(f)
    
    # 加载时间模型归一化参数
    with open('arf_model_t_normalization_params.json', 'r') as f:
        norm_params_t = json.load(f)
    
    return model_x, model_t, norm_params_x, norm_params_t


def predict_spatial_factor(model_x, norm_params_x, x_values_mm):
    """使用空间模型预测空间因子"""
    model_x.eval()
    with torch.no_grad():
        # 将毫米转换为米
        x_values_m = x_values_mm / 1000.0
        
        # 归一化输入
        x_min_m = norm_params_x['x_min']  # 米
        x_max_m = norm_params_x['x_max']  # 米
        x_scaled = (x_values_m - x_min_m) / (x_max_m - x_min_m)
        
        # 模型预测
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)
        pred_norm = model_x(x_tensor)
        
        # 反归一化输出
        force_mu = norm_params_x['force_mu']
        force_sigma = norm_params_x['force_sigma']
        pred_factor = pred_norm * force_sigma + force_mu
        
        return pred_factor.numpy().flatten()


def predict_temporal_factor(model_t, norm_params_t, t_values_us):
    """使用时间模型预测时间因子"""
    model_t.eval()
    with torch.no_grad():
        # 将微秒转换为秒
        t_values_s = t_values_us / 1e6
        
        # 归一化输入
        t_min_s = norm_params_t['t_min']  # 秒
        t_max_s = norm_params_t['t_max']  # 秒
        t_scaled = (t_values_s - t_min_s) / (t_max_s - t_min_s)
        
        # 模型预测
        t_tensor = torch.tensor(t_scaled, dtype=torch.float32).unsqueeze(1)
        pred_norm = model_t(t_tensor)
        
        # 反归一化输出
        time_factor_mu = norm_params_t['time_factor_mu']
        time_factor_sigma = norm_params_t['time_factor_sigma']
        pred_factor = pred_norm * time_factor_sigma + time_factor_mu
        
        return pred_factor.numpy().flatten()


def theoretical_spatial_factor(x_values_mm, particle_radius=1e-6):
    """计算理论空间因子（t=0时的力）"""
    # 将毫米转换为米
    x_values_m = x_values_mm / 1000.0
    x_tensor = torch.tensor(x_values_m, dtype=torch.float32).unsqueeze(1)
    
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    
    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x_tensor - offset
    x_back = x_tensor + offset
    
    # t=0时，cos(ωt) = cos(0) = 1
    p_front = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_front) / wavelength
    p_back = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_back) / wavelength
    
    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4.0
    return force_magnitude.numpy().flatten()


def theoretical_temporal_factor(t_values_us):
    """计算理论时间因子：cos(ωt)"""
    t_values_s = t_values_us / 1e6
    omega = 2 * np.pi * frequency
    return np.cos(omega * t_values_s)


def theoretical_arf_full(x_values_mm, t_values_us, particle_radius=1e-6):
    """计算完整的理论ARF：F(x,t) = F_spatial(x) * cos(ωt)"""
    spatial_factor = theoretical_spatial_factor(x_values_mm, particle_radius)
    temporal_factor = theoretical_temporal_factor(t_values_us)
    return spatial_factor * temporal_factor

def plot_comparison():
    """绘制分离式PINN模型预测与理论值的对比图"""
    # 加载模型和参数
    model_x, model_t, norm_params_x, norm_params_t = load_models_and_params()
    
    # 设置绘图范围
    x_min_mm = -25.5  # 毫米
    x_max_mm = 25.5   # 毫米
    t_min_us = 0.0    # 微秒
    t_max_us = 2.0    # 微秒（2个声波周期）
    
    print(f"绘图范围: x=[{x_min_mm:.1f}, {x_max_mm:.1f}] mm, t=[{t_min_us:.1f}, {t_max_us:.1f}] μs")
    
    # 生成测试数据点
    x_range_mm = np.linspace(x_min_mm, x_max_mm, 200)
    t_range_us = np.linspace(t_min_us, t_max_us, 200)
    
    # 计算理论值
    print("计算理论空间因子...")
    theoretical_spatial = theoretical_spatial_factor(x_range_mm)
    
    print("计算理论时间因子...")
    theoretical_temporal = theoretical_temporal_factor(t_range_us)
    
    # 计算模型预测值
    print("计算PINN预测空间因子...")
    predicted_spatial = predict_spatial_factor(model_x, norm_params_x, x_range_mm)
    
    print("计算PINN预测时间因子...")
    predicted_temporal = predict_temporal_factor(model_t, norm_params_t, t_range_us)
    
    # 计算相对误差
    spatial_errors = np.abs(predicted_spatial - theoretical_spatial) / (np.abs(theoretical_spatial) + 1e-30)
    temporal_errors = np.abs(predicted_temporal - theoretical_temporal) / (np.abs(theoretical_temporal) + 1e-30)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 空间因子对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(x_range_mm, theoretical_spatial * 1e12, 'b-', label='理论值', linewidth=2)
    ax1.plot(x_range_mm, predicted_spatial * 1e12, 'r--', label='PINN预测', linewidth=2)
    ax1.set_xlabel('位置 x (mm)')
    ax1.set_ylabel('空间因子 F(x) (pN)')
    ax1.set_title('空间因子对比 (t=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 时间因子对比
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(t_range_us, theoretical_temporal, 'b-', label='理论值', linewidth=2)
    ax2.plot(t_range_us, predicted_temporal, 'r--', label='PINN预测', linewidth=2)
    ax2.set_xlabel('时间 t (μs)')
    ax2.set_ylabel('时间因子 cos(ωt)')
    ax2.set_title('时间因子对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 空间因子误差
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(x_range_mm, spatial_errors * 100, 'g-', linewidth=2)
    ax3.set_xlabel('位置 x (mm)')
    ax3.set_ylabel('相对误差 (%)')
    ax3.set_title('空间因子相对误差')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(1e-3, 1e2)
    
    # 4. 时间因子误差
    ax4 = plt.subplot(2, 3, 4)
    ax4.semilogy(t_range_us, temporal_errors * 100, 'g-', linewidth=2)
    ax4.set_xlabel('时间 t (μs)')
    ax4.set_ylabel('相对误差 (%)')
    ax4.set_title('时间因子相对误差')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(1e-3, 1e2)
    
    # 5. 完整ARF的时空变化（热图）
    ax5 = plt.subplot(2, 3, (5, 6))
    X_mesh, T_mesh = np.meshgrid(x_range_mm, t_range_us)
    
    # 计算完整ARF
    theoretical_full = theoretical_arf_full(X_mesh, T_mesh)
    predicted_full = predicted_spatial * predicted_temporal.reshape(-1, 1)
    
    # 绘制理论值热图
    im = ax5.contourf(X_mesh, T_mesh, theoretical_full * 1e12, levels=20, cmap='RdBu_r')
    ax5.set_xlabel('位置 x (mm)')
    ax5.set_ylabel('时间 t (μs)')
    ax5.set_title('完整ARF时空变化 (理论值)')
    plt.colorbar(im, ax=ax5, label='ARF (pN)')
    
    plt.tight_layout()
    plt.savefig('arf_pinn_separated_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== 空间因子统计 ===")
    print(f"平均相对误差: {np.mean(spatial_errors) * 100:.2f}%")
    print(f"最大相对误差: {np.max(spatial_errors) * 100:.2f}%")
    print(f"最小相对误差: {np.min(spatial_errors) * 100:.2f}%")
    
    print("\n=== 时间因子统计 ===")
    print(f"平均相对误差: {np.mean(temporal_errors) * 100:.2f}%")
    print(f"最大相对误差: {np.max(temporal_errors) * 100:.2f}%")
    print(f"最小相对误差: {np.min(temporal_errors) * 100:.2f}%")
    
    # 打印关键点对比
    print("\n=== 空间因子关键点对比 ===")
    key_points_mm = [x_min_mm, x_min_mm/2, 0.0, x_max_mm/2, x_max_mm]
    print("位置(mm) | 理论值(pN) | PINN预测(pN) | 相对误差(%)")
    print("-" * 60)
    for x_mm in key_points_mm:
        idx = np.argmin(np.abs(x_range_mm - x_mm))
        theo = theoretical_spatial[idx] * 1e12
        pred = predicted_spatial[idx] * 1e12
        rel_err = spatial_errors[idx] * 100
        print(f"{x_mm:8.1f} | {theo:10.2e} | {pred:10.2e} | {rel_err:8.2f}")
    
    print("\n=== 时间因子关键点对比 ===")
    period_us = 1e6 / frequency  # 声波周期（微秒）
    key_times_us = [0, period_us/4, period_us/2, 3*period_us/4, period_us]
    print("时间(μs) | 理论值 | PINN预测 | 相对误差(%)")
    print("-" * 50)
    for t_us in key_times_us:
        idx = np.argmin(np.abs(t_range_us - t_us))
        theo = theoretical_temporal[idx]
        pred = predicted_temporal[idx]
        rel_err = temporal_errors[idx] * 100
        print(f"{t_us:8.1f} | {theo:7.4f} | {pred:9.4f} | {rel_err:8.2f}")

if __name__ == "__main__":
    try:
        plot_comparison()
        print("\nComparison plot saved as 'arf_pinn_separated_comparison.png'")
    except FileNotFoundError as e:
        print(f"Error: File not found {e}")
        print("Please ensure the trained model files exist:")
        print("- arf_model_x.pth")
        print("- arf_model_x_normalization_params.json")
        print("- arf_model_t.pth")
        print("- arf_model_t_normalization_params.json")
    except Exception as e:
        print(f"Error occurred: {e}")