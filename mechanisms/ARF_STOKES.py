import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

# 导入物理参数和函数
from initialization.sound_source_standing import frequency, sound_pressure_level, spl_to_pressure
from mechanisms.ARF import p_0, rho_0, gamma, c_0, get_amplitude
from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound

# 物理参数
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa


class Normalizer:
    """
    归一化器，用于输入输出数据的标准化
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in ['x', 'vx', 't', 'fx']:
            mean_key = f'{k}_mean'
            std_key = f'{k}_std'
            if mean_key in data and std_key in data:
                self.stats[k] = (
                    torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                    torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                )

    def transform(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = (v - mu) / sigma
            else:
                out[k] = v
        return out

    def inverse(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = v * sigma + mu
            else:
                out[k] = v
        return out


class ARFStokesNet(nn.Module):
    """
    统一的ARF+Stokes力预测网络
    输入: [x, vx, t] (位置x, 速度vx, 时间t)
    输出: [Fx] (x方向合力)
    """
    def __init__(self, fourier_features=16):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算波数用于傅里叶特征
        wavelength = c_0 / frequency  # 0.034 m
        k = 2 * np.pi / wavelength    # 约 184.8 rad/m
        
        # 傅里叶特征权重（用于位置和时间的周期性）
        self.register_buffer('fourier_weights_x', torch.randn(1, fourier_features) * k)
        self.register_buffer('fourier_weights_t', torch.randn(1, fourier_features) * 2 * np.pi * frequency)
        
        # 主网络
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 4 + 3, 256),  # 位置sin/cos + 时间sin/cos + x + vx + t
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # 输出 Fx
        )

    def forward(self, x, vx, t):
        # 生成位置相关的傅里叶特征
        fourier_x = self.fourier_weights_x * x
        fourier_features_x = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        
        # 生成时间相关的傅里叶特征
        fourier_t = self.fourier_weights_t * t
        fourier_features_t = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        
        # 组合所有特征
        combined_input = torch.cat([fourier_features_x, fourier_features_t, x, vx, t], dim=1)
        
        # 通过主网络
        return self.layers(combined_input)


def theoretical_arf_stokes_force(x, vx, t, particle_radius=1e-6):
    """
    计算理论ARF+Stokes合力（仅x方向）
    
    ARF力: F_arf = πd_p²[p(x-√2d_p/4,t) - p(x+√2d_p/4,t)] / 4
    Stokes力: F_stokes = -3πμd_p(vx - ux)/C_c
    
    其中:
    - p(x,t) = 2πAρ₀γsin(2πx/λ)cos(2πft)/λ
    - ux = -2πfA cos(2πx/λ)sin(2πft)
    """
    # 粒子直径
    d_p = 2 * particle_radius
    
    # 计算波长和波数
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    
    # 计算振幅A
    sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
    A = sound_pressure / (rho_0 * c_0 * 2 * np.pi * frequency)
    
    # 计算ARF力
    offset = np.sqrt(2) * d_p / 4
    x_front = x - offset
    x_back = x + offset
    
    # 声压计算
    p_front = 2 * 2 * np.pi * A * p_0 * gamma * torch.sin(k * x_front) * torch.cos(omega * t) / wavelength
    p_back = 2 * 2 * np.pi * A * p_0 * gamma * torch.sin(k * x_back) * torch.cos(omega * t) / wavelength
    
    # ARF力
    pressure_diff = p_front - p_back
    F_arf = np.pi * d_p**2 * pressure_diff / 4
    
    # 计算Stokes力
    # 空气速度: ux = -2πfA cos(2πx/λ)sin(2πft)
    ux = -2 * np.pi * frequency * A * torch.cos(k * x) * torch.sin(omega * t)
    
    # 相对速度
    relative_velocity = vx - ux
    
    # Stokes阻力系数
    drag_coeff = 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham
    F_stokes = -drag_coeff * relative_velocity
    
    # 总力
    F_total = F_arf + F_stokes
    
    return F_total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("基于ARF+Stokes解析公式的统一PINN训练…")
    print(f"device: {device}")

    # 数据生成
    N = 50000  # 增加数据量以提高训练效果
    
    # 位置范围：覆盖3个波长
    wavelength = c_0 / frequency
    x_range = 3 * wavelength
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()
    
    # 速度范围：[-0.1, 0.1] m/s
    vx = (torch.rand(N, 1, device=device) * 0.2 - 0.1).float()
    
    # 时间范围：覆盖2个周期
    period = 1.0 / frequency
    t_range = 2 * period
    t = (torch.rand(N, 1, device=device) * t_range).float()
    
    # 计算理论力
    F_theory = theoretical_arf_stokes_force(x, vx, t).detach()
    
    # 特征缩放
    x_min, x_max = -x_range/2, x_range/2
    vx_min, vx_max = -0.1, 0.1
    t_min, t_max = 0.0, t_range
    
    x_norm = (x - x_min) / (x_max - x_min)
    vx_norm = (vx - vx_min) / (vx_max - vx_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # 输出归一化
    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    F_norm = (F_theory - force_mu) / force_sigma
    
    # 模型/优化器
    model = ARFStokesNet(fourier_features=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000)
    
    # 训练
    target_loss = 1e-2
    print_interval = 1000
    epoch = 0
    losses = []
    
    print("开始训练… 目标损失:", target_loss)
    print(f"采样范围: x=[{x_min:.3f}, {x_max:.3f}] m, vx=[{vx_min:.3f}, {vx_max:.3f}] m/s, t=[{t_min:.2e}, {t_max:.2e}] s")
    
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(x_norm, vx_norm, t_norm)
        loss = criterion(pred, F_norm)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())
        
        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        if loss.item() < target_loss:
            print(f"✅ 达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        epoch += 1
    
    # 可视化训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ARF+Stokes Unified Model Training Loss')
    plt.grid(True)
    plt.show()
    
    # 测试
    model.eval()
    with torch.no_grad():
        # 测试点
        test_x = torch.linspace(x_min, x_max, 50, device=device).unsqueeze(1)
        test_vx = torch.linspace(vx_min, vx_max, 10, device=device).unsqueeze(1)
        test_t = torch.linspace(t_min, t_max, 20, device=device).unsqueeze(1)
        
        # 创建测试网格
        test_x_grid, test_vx_grid, test_t_grid = torch.meshgrid(
            test_x.squeeze(), test_vx.squeeze(), test_t.squeeze(), indexing='ij'
        )
        test_x_flat = test_x_grid.flatten().unsqueeze(1)
        test_vx_flat = test_vx_grid.flatten().unsqueeze(1)
        test_t_flat = test_t_grid.flatten().unsqueeze(1)
        
        # 归一化测试数据
        test_x_norm = (test_x_flat - x_min) / (x_max - x_min)
        test_vx_norm = (test_vx_flat - vx_min) / (vx_max - vx_min)
        test_t_norm = (test_t_flat - t_min) / (t_max - t_min)
        
        # 预测
        pred_norm = model(test_x_norm, test_vx_norm, test_t_norm)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_arf_stokes_force(test_x_flat, test_vx_flat, test_t_flat)
        
        # 计算误差
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {mean_error*100:.2f}%")
        print(f"最大相对误差: {max_error*100:.2f}%")
        
        # 绘制一些测试结果
        # 固定vx=0, t=0，绘制F(x)
        vx_zero = torch.zeros_like(test_x)
        t_zero = torch.zeros_like(test_x)
        vx_zero_norm = (vx_zero - vx_min) / (vx_max - vx_min)
        t_zero_norm = (t_zero - t_min) / (t_max - t_min)
        
        # 确保所有测试张量维度一致
        test_x_norm_fixed = (test_x - x_min) / (x_max - x_min)
        
        pred_fx = model(test_x_norm_fixed, vx_zero_norm, t_zero_norm)
        pred_fx_denorm = pred_fx * force_sigma + force_mu
        true_fx = theoretical_arf_stokes_force(test_x, vx_zero, t_zero)
        
        plt.figure(figsize=(12, 5))
        
        # 子图1：F(x) at vx=0, t=0
        plt.subplot(1, 2, 1)
        x_mm = (test_x * 1000).detach().cpu().numpy().flatten()
        pred_pn = (pred_fx_denorm * 1e12).detach().cpu().numpy().flatten()
        true_pn = (true_fx * 1e12).detach().cpu().numpy().flatten()
        plt.plot(x_mm, true_pn, 'b-', label='Theory', linewidth=2)
        plt.plot(x_mm, pred_pn, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Position x (mm)')
        plt.ylabel('Force Fx (pN)')
        plt.title('F(x) at vx=0, t=0')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 子图2：F(t) at x=0, vx=0
        plt.subplot(1, 2, 2)
        x_zero = torch.zeros_like(test_t)
        vx_zero_t = torch.zeros_like(test_t)
        x_zero_norm = (x_zero - x_min) / (x_max - x_min)
        vx_zero_t_norm = (vx_zero_t - vx_min) / (vx_max - vx_min)
        
        # 确保时间测试张量维度一致
        test_t_norm_fixed = (test_t - t_min) / (t_max - t_min)
        
        pred_ft = model(x_zero_norm, vx_zero_t_norm, test_t_norm_fixed)
        pred_ft_denorm = pred_ft * force_sigma + force_mu
        true_ft = theoretical_arf_stokes_force(x_zero, vx_zero_t, test_t)
        
        t_us = (test_t * 1e6).detach().cpu().numpy().flatten()
        pred_ft_pn = (pred_ft_denorm * 1e12).detach().cpu().numpy().flatten()
        true_ft_pn = (true_ft * 1e12).detach().cpu().numpy().flatten()
        plt.plot(t_us, true_ft_pn, 'b-', label='Theory', linewidth=2)
        plt.plot(t_us, pred_ft_pn, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Time (μs)')
        plt.ylabel('Force Fx (pN)')
        plt.title('F(t) at x=0, vx=0')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # 保存模型与归一化参数
    torch.save(model.state_dict(), 'PINN/arf_stokes_unified_model.pth')
    with open('PINN/arf_stokes_unified_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'vx_min': vx_min, 'vx_max': vx_max,
            't_min': t_min, 't_max': t_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("统一ARF+Stokes模型和归一化参数已保存至 PINN/。")


if __name__ == "__main__":
    main()
