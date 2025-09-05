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

import re

class SimpleMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SpatialWrapper(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        wavelength = c_0 / frequency
        self.register_buffer('k', torch.tensor(2 * np.pi / wavelength, dtype=torch.float32))

    def forward(self, x_m):
        phase = self.k * x_m
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.mlp(feats)

class TemporalWrapper(nn.Module):
    def __init__(self, mlp, t_min_s, t_max_s):
        super().__init__()
        self.mlp = mlp
        self.register_buffer('t_min', torch.tensor(float(t_min_s), dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(float(t_max_s), dtype=torch.float32))
        self.register_buffer('omega', torch.tensor(2 * np.pi * frequency, dtype=torch.float32))

    def forward(self, t_norm):
        t_sec = t_norm * (self.t_max - self.t_min) + self.t_min
        phase = self.omega * t_sec
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.mlp(feats)


def _infer_layer_dims_from_state(state_dict):
    linear_keys = sorted([k for k in state_dict.keys() if re.match(r"layers\.[0-9]+\.weight", k)])
    if not linear_keys:
        return None
    dims = []
    for i, k in enumerate(linear_keys):
        W = state_dict[k]
        out_dim, in_dim = W.shape
        if i == 0:
            dims.append(in_dim)
        dims.append(out_dim)
    return dims

def load_models_and_params():
    """加载训练好的空间和时间模型及其归一化参数"""
    # 加载空间模型（优先从 PINN/ 目录）
    try:
        state_x = torch.load('PINN/arf_model_x.pth', map_location='cpu')
    except FileNotFoundError:
        state_x = torch.load('arf_model_x.pth', map_location='cpu')
    # 兼容多种保存格式
    if isinstance(state_x, dict) and 'model_state_dict' in state_x:
        state_x = state_x['model_state_dict']
    elif isinstance(state_x, dict) and 'state_dict' in state_x:
        state_x = state_x['state_dict']
    # 动态构建与checkpoint匹配的MLP
    dims_x = _infer_layer_dims_from_state(state_x)
    if dims_x is None:
        dims_x = [2, 256, 256, 128, 64, 1]
    mlp_x = SimpleMLP(dims_x)
    model_x = SpatialWrapper(mlp_x)
    model_x.load_state_dict(state_x, strict=False)
    model_x.eval()
    
    # 加载时间模型
    try:
        state_t = torch.load('PINN/arf_model_t.pth', map_location='cpu')
    except FileNotFoundError:
        state_t = torch.load('arf_model_t.pth', map_location='cpu')
    # 兼容多种保存格式
    if isinstance(state_t, dict) and 'model_state_dict' in state_t:
        state_t = state_t['model_state_dict']
    elif isinstance(state_t, dict) and 'state_dict' in state_t:
        state_t = state_t['state_dict']
    dims_t = _infer_layer_dims_from_state(state_t)
    if dims_t is None:
        dims_t = [2, 256, 256, 128, 64, 1]
    # 读取时间归一化以便从 t_norm 还原到秒
    try:
        with open('PINN/arf_model_t_normalization_params.json', 'r') as f:
            norm_t_preview = json.load(f)
    except FileNotFoundError:
        with open('arf_model_t_normalization_params.json', 'r') as f:
            norm_t_preview = json.load(f)
    mlp_t = SimpleMLP(dims_t)
    model_t = TemporalWrapper(mlp_t, norm_t_preview['t_min'], norm_t_preview['t_max'])
    model_t.load_state_dict(state_t, strict=False)
    model_t.eval()
    
    # 加载空间模型归一化参数（多重回退）
    try:
        with open('PINN/arf_model_x_normalization_params.json', 'r') as f:
            norm_params_x = json.load(f)
    except FileNotFoundError:
        try:
            with open('arf_model_x_normalization_params.json', 'r') as f:
                norm_params_x = json.load(f)
        except FileNotFoundError:
            # 进一步回退到旧命名
            try:
                with open('PINN/arf_pinn_normalization_params.json', 'r') as f:
                    norm_params_x = json.load(f)
            except FileNotFoundError:
                try:
                    with open('arf_pinn_normalization_params.json', 'r') as f:
                        norm_params_x = json.load(f)
                except FileNotFoundError:
                    # 最终回退：基于理论默认范围构建参数
                    wavelength = c_0 / frequency
                    half_period = wavelength / 2.0
                    x_min_m = -1.5 * half_period
                    x_max_m =  1.5 * half_period
                    # 用理论力估算 mu/sigma（训练也是用理论力标准化）
                    xs = np.linspace(x_min_m, x_max_m, 1000, dtype=np.float32)
                    xs_t = torch.tensor(xs).unsqueeze(1)
                    theo = theoretical_spatial_factor(xs * 1000.0)  # 传毫米
                    force_mu = float(np.mean(theo))
                    force_sigma = float(max(np.std(theo), 1e-30))
                    norm_params_x = {
                        'x_min': x_min_m,
                        'x_max': x_max_m,
                        'force_mu': force_mu,
                        'force_sigma': force_sigma,
                    }
    
    # 加载时间模型归一化参数（若上面已读到预览则直接使用；否则构建默认）
    norm_params_t = norm_t_preview
    if norm_params_t is None:
        try:
            with open('PINN/arf_model_t_normalization_params.json', 'r') as f:
                norm_params_t = json.load(f)
        except FileNotFoundError:
            try:
                with open('arf_model_t_normalization_params.json', 'r') as f:
                    norm_params_t = json.load(f)
            except FileNotFoundError:
                # 默认时间范围一个周期，mu/sigma 用理论 cos 计算
                t_min_s = 0.0
                t_max_s = 1.0 / frequency
                ts = np.linspace(t_min_s, t_max_s, 2000, dtype=np.float32)
                time_mu = float(np.mean(np.cos(2 * np.pi * frequency * ts)))
                time_sigma = float(max(np.std(np.cos(2 * np.pi * frequency * ts)), 1e-30))
                norm_params_t = {
                    't_min': t_min_s,
                    't_max': t_max_s,
                    'time_factor_mu': time_mu,
                    'time_factor_sigma': time_sigma,
                }
    
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
    """计算理论空间因子（t=0时的力），兼容1D或2D输入，返回同形状的numpy数组"""
    # 将毫米转换为米（保持输入形状）
    x_values_m = np.asarray(x_values_mm, dtype=np.float32) / 1000.0

    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength

    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x_values_m - offset
    x_back = x_values_m + offset

    # t=0时，cos(ωt) = cos(0) = 1
    p_front = 2.0 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_front) / wavelength
    p_back = 2.0 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_back) / wavelength

    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4.0
    return force_magnitude


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
    # predicted_spatial: (200,), predicted_temporal: (200,)
    predicted_full = np.outer(predicted_temporal, predicted_spatial)
    
    # 绘制理论值热图
    if theoretical_full.ndim == 3:
        theoretical_full = theoretical_full.squeeze()
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