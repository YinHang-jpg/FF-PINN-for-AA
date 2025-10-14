import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# 兼容导入 Stokes_drag.py 中的常量与解析函数
try:
    from .Stokes_drag import compute_air_velocity_due_to_sound
except Exception:
    from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound

# 物理参数
frequency = 10000  # Hz
sound_pressure_level = 140  # dB
sound_speed = 340  # m/s
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa
rho_0 = 1.225  # kg/m³
c_0 = 340  # m/s


class Normalizer:
    """
    轻量级归一化器，供 main.py 安全导入使用。
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in ['x', 'vx', 'vy', 't', 'fx', 'fy']:
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


class StokesNetX(nn.Module):
    """
    周期性感知的MLP：输入 [x, vx, vy, t]，输出 [Fx, Fy]
    使用傅里叶特征来更好地捕捉声波的周期性特征
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算波数用于傅里叶特征
        wavelength = sound_speed / frequency  # 0.034 m
        k = 2 * np.pi / wavelength  # 约 184.8 rad/m
        
        # 傅里叶特征权重（多个频率分量）
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
        
        # 主网络（大幅简化）
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 1, 32),  # sin/cos特征 + x（与已训练权重一致）
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # 输出 Fx
        )

    def forward(self, x):
        # 生成傅里叶特征（基于位置x）
        fourier_x = self.fourier_weights * x
        fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        # 与已训练结构对齐：拼接归一化 x 通道
        combined_input = torch.cat([fourier_features, x], dim=1)
        
        # 通过主网络
        return self.layers(combined_input)


def theoretical_stokes_force_x(x, particle_radius=1e-6):
    """
    基于 Stokes_drag.py 中的物理公式计算理论斯托克斯阻力（仅x方向，位置相关）：
    只返回位置因子 cos(2πx/λ)，其他系数在组合时处理
    """
    # 计算波长和波数
    wavelength = sound_speed / frequency
    k = 2 * np.pi / wavelength
    
    # 只返回位置相关的系数：cos(2πx/λ)
    position_factor = torch.cos(k * x)
    
    return position_factor


def main():
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("基于 Stokes 解析公式的 PINN 训练（位置相关）…")
    print(f"device: {device}")

    # 数据生成（仅位置x）
    wavelength = sound_speed / frequency  # 0.034 m
    x_range = 3 * wavelength  # 覆盖3个波长
    
    N = 20000
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()  # [-0.051, 0.051] m

    F_theory = theoretical_stokes_force_x(x).detach()

    # 特征缩放（仅输入x）
    x_min, x_max = -x_range/2, x_range/2
    x_scaled = (x - x_min) / (x_max - x_min)
    X_theory = x_scaled

    # 归一化输出（仅x方向力）
    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    Y_theory = (F_theory - force_mu) / force_sigma

    # 模型/优化器
    model = StokesNetX(fourier_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # 训练直到达到目标损失
    target_loss = 2.5e-5
    print_interval = 500
    epoch = 0
    losses = []

    print("开始训练… 目标损失:", target_loss)
    print("使用傅里叶特征增强周期性学习能力")
    print(f"采样范围: x=[{x_min:.3f}, {x_max:.3f}] m")
    
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(X_theory)
        loss = criterion(pred, Y_theory)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())

        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        if loss.item() < target_loss:
            print(f"达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        epoch += 1

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    # 详细测试
    model.eval()
    with torch.no_grad():
        # 测试点
        test_x = torch.linspace(x_min, x_max, 50, device=device).unsqueeze(1)
        test_x_scaled = (test_x - x_min) / (x_max - x_min)
        
        pred_norm = model(test_x_scaled)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_stokes_force_x(test_x)
        
        # 计算整体误差统计
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {mean_error*100:.2f}%")
        print(f"最大相对误差: {max_error*100:.2f}%")

        # 绘制 F(x) 理论值 与 模型预测 曲线
        x_mm = (test_x * 1000.0).detach().cpu().numpy().flatten()
        pred_fx = (pred_force * 1e12).detach().cpu().numpy().flatten()
        true_fx = (true_force * 1e12).detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 5))
        plt.plot(x_mm, true_fx, 'b-', label='Theory', linewidth=2)
        plt.plot(x_mm, pred_fx, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Position x (mm)')
        plt.ylabel('Stokes Force Fx (pN)')
        plt.title('Fx(x) Theory vs PINN')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 保存模型与归一化参数
    torch.save(model.state_dict(), 'PINN/stokes_model_x.pth')
    with open('PINN/stokes_model_x_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("位置相关斯托克斯阻力模型和归一化参数已保存。")


if __name__ == "__main__":
    main()
