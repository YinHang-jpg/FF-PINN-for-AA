import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from pathlib import Path

# 导入物理常量
try:
    import sys
    sys.path.append('..')
    from mechanisms.ARF import p_0, rho_0, gamma, c_0
except Exception:
    # 默认物理常量
    p_0 = 101325  # Pa
    rho_0 = 1.225  # kg/m³
    gamma = 1.4
    c_0 = 340  # m/s

# 参考压力（用于SPL计算）
reference_pressure = 20e-6  # Pa
viscosity = 1.8e-5  # Pa·s
particle_radius = 1e-6  # m
particle_density = 1000.0  # kg/m³ (固定密度)
cunningham = 1.083

# 计算弛豫时间
d_p = 2.0 * particle_radius
m_p = particle_density * (4.0/3.0) * np.pi * particle_radius**3
drag_coeff = 3 * np.pi * viscosity * d_p / cunningham
t_relaxation = m_p / drag_coeff  # 粒子弛豫时间


class Normalizer:
    """归一化器"""
    def __init__(self):
        self.stats = {}
    
    def fit(self, data_dict):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                mu = v.mean()
                sigma = v.std().clamp_min(1e-30)
                self.stats[k] = (mu, sigma)
    
    def transform(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = (v - mu) / sigma
            else:
                out[k] = v
        return out
    
    def inverse_transform(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = v * sigma + mu
            else:
                out[k] = v
        return out
    
    def save(self, path):
        data = {}
        for k, (mu, sigma) in self.stats.items():
            data[f'{k}_mean'] = float(mu.item()) if isinstance(mu, torch.Tensor) else float(mu)
            data[f'{k}_std'] = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in list(data.keys()):
            if k.endswith('_mean'):
                base_key = k[:-5]
                std_key = f'{base_key}_std'
                if std_key in data:
                    self.stats[base_key] = (
                        torch.tensor(data[k], device=device, dtype=torch.float32),
                        torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                    )


# ==================== 模块1: ARF空间分量 ====================
class ARFNetX(nn.Module):
    """声辐射力的空间分量: f_ARF_x(x, SPL, freq)"""
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        self.register_buffer('spatial_weights', torch.randn(1, fourier_features) * 200)
        
        # 输入: 傅里叶特征(sin/cos) + x + SPL + freq (更深更宽的网络)
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, SPL, freq):
        fourier_x = self.spatial_weights * x
        fourier_spatial = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        features = torch.cat([fourier_spatial, x, SPL, freq], dim=1)
        return self.layers(features)


# ==================== 模块2: ARF时间分量 ====================
class ARFNetT(nn.Module):
    """声辐射力的时间分量: f_ARF_t(t, freq)"""
    def __init__(self, fourier_features=16):
        super().__init__()
        self.fourier_features = fourier_features
        self.register_buffer('temporal_weights', torch.randn(1, fourier_features) * 2e4)
        
        # 输入: 傅里叶特征(sin/cos) + t + freq (更深的网络)
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t, freq):
        fourier_t = self.temporal_weights * t
        fourier_temporal = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        features = torch.cat([fourier_temporal, t, freq], dim=1)
        return self.layers(features)


# ==================== 模块3: Stokes空间分量 ====================
class StokesNetX(nn.Module):
    """斯托克斯阻力的空间分量: f_Stokes_x(x, SPL, freq)"""
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        self.register_buffer('spatial_weights', torch.randn(1, fourier_features) * 200)
        
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x, SPL, freq):
        fourier_x = self.spatial_weights * x
        fourier_spatial = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        features = torch.cat([fourier_spatial, x, SPL, freq], dim=1)
        return self.layers(features)


# ==================== 模块4: Stokes速度分量 ====================
class StokesNetV(nn.Module):
    """斯托克斯阻力的速度分量: f_Stokes_v(vx)"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
    
    def forward(self, vx):
        return self.layers(vx)


# ==================== 模块5: Stokes时间分量 ====================
class StokesNetT(nn.Module):
    """斯托克斯阻力的时间分量: f_Stokes_t(t, freq)"""
    def __init__(self, fourier_features=16):
        super().__init__()
        self.fourier_features = fourier_features
        self.register_buffer('temporal_weights', torch.randn(1, fourier_features) * 2e4)
        
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t, freq):
        fourier_t = self.temporal_weights * t
        fourier_temporal = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        features = torch.cat([fourier_temporal, t, freq], dim=1)
        return self.layers(features)


# ==================== 组合模型 ====================
class CombinedForceModel(nn.Module):
    """
    组合模型：通过物理公式组合五个子模块
    F_total = F_ARF + F_Stokes
    F_ARF = f_ARF_x(x, SPL, freq) * f_ARF_t(t, freq)
    F_Stokes = f_Stokes_x(x, SPL, freq) * f_Stokes_v(vx) * f_Stokes_t(t, freq)
    """
    def __init__(self, arf_x, arf_t, stokes_x, stokes_v, stokes_t):
        super().__init__()
        self.arf_x = arf_x
        self.arf_t = arf_t
        self.stokes_x = stokes_x
        self.stokes_v = stokes_v
        self.stokes_t = stokes_t
    
    def forward(self, x, t, vx, SPL, freq):
        # ARF部分：位置 × 时间
        f_arf_x = self.arf_x(x, SPL, freq)
        f_arf_t = self.arf_t(t, freq)
        F_arf = f_arf_x * f_arf_t
        
        # Stokes部分：位置 × 速度 × 时间
        f_stokes_x = self.stokes_x(x, SPL, freq)
        f_stokes_v = self.stokes_v(vx)
        f_stokes_t = self.stokes_t(t, freq)
        
        F_stokes = f_stokes_x * f_stokes_v * f_stokes_t
        
        # 总力
        F_total = F_arf + F_stokes
        
        return F_total


# ==================== 理论公式 ====================
def theoretical_arf_x(x, SPL, freq):
    """ARF空间分量理论值"""
    P_rms = reference_pressure * 10 ** (SPL / 20.0)
    P_amplitude = P_rms * np.sqrt(2)
    wavelength = c_0 / freq
    k = 2 * np.pi / wavelength
    
    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    area = np.pi * particle_radius ** 2
    
    pressure_gradient = P_amplitude * k * (
        torch.cos(k * (x - offset)) - torch.cos(k * (x + offset))
    )
    
    return area * pressure_gradient


def theoretical_arf_t(t, freq):
    """ARF时间分量理论值"""
    omega = 2 * np.pi * freq
    return torch.cos(omega * t)


def theoretical_stokes_x(x, SPL, freq):
    """Stokes空间分量理论值"""
    P_rms = reference_pressure * 10 ** (SPL / 20.0)
    P_amplitude = P_rms * np.sqrt(2)
    wavelength = c_0 / freq
    k = 2 * np.pi / wavelength
    
    # 空气速度振幅的空间分布
    v_air_amplitude = P_amplitude / (rho_0 * c_0)
    position_factor = torch.cos(k * x)
    
    return drag_coeff * v_air_amplitude * position_factor


def theoretical_stokes_v(vx):
    """Stokes速度分量理论值"""
    d_p = 2.0 * particle_radius
    drag_coeff = 3 * np.pi * viscosity * d_p / cunningham
    return -drag_coeff * vx


def theoretical_stokes_t(t, freq):
    """Stokes时间分量理论值"""
    omega = 2 * np.pi * freq
    # 时间项本身的周期性变化
    return torch.cos(omega * t)


# ==================== 数据生成 ====================
def generate_data_for_module(module_name, N, device):
    """为每个模块生成训练数据"""
    if module_name == 'arf_x':
        # 采样范围
        SPL = torch.rand(N, 1, device=device) * 40 + 120  # 120-160 dB
        freq = torch.rand(N, 1, device=device) * 15000 + 5000  # 5-20 kHz
        
        # 位置范围：根据频率确定波长
        x_list = []
        for i in range(N):
            wavelength = c_0 / freq[i].item()
            x_range = 3 * wavelength
            x_i = (torch.rand(1, device=device) * x_range - x_range/2).item()
            x_list.append(x_i)
        x = torch.tensor(x_list, device=device).unsqueeze(1).float()
        
        y = theoretical_arf_x(x, SPL, freq).detach()
        return {'x': x, 'SPL': SPL, 'freq': freq, 'y': y}
    
    elif module_name == 'arf_t':
        freq = torch.rand(N, 1, device=device) * 15000 + 5000  # 5-20 kHz
        
        # 时间范围：根据频率确定周期
        t_list = []
        for i in range(N):
            period = 1.0 / freq[i].item()
            t_i = (torch.rand(1, device=device) * period).item()
            t_list.append(t_i)
        t = torch.tensor(t_list, device=device).unsqueeze(1).float()
        
        y = theoretical_arf_t(t, freq).detach()
        return {'t': t, 'freq': freq, 'y': y}
    
    elif module_name == 'stokes_x':
        SPL = torch.rand(N, 1, device=device) * 40 + 120
        freq = torch.rand(N, 1, device=device) * 15000 + 5000
        
        x_list = []
        for i in range(N):
            wavelength = c_0 / freq[i].item()
            x_range = 3 * wavelength
            x_i = (torch.rand(1, device=device) * x_range - x_range/2).item()
            x_list.append(x_i)
        x = torch.tensor(x_list, device=device).unsqueeze(1).float()
        
        y = theoretical_stokes_x(x, SPL, freq).detach()
        return {'x': x, 'SPL': SPL, 'freq': freq, 'y': y}
    
    elif module_name == 'stokes_v':
        vx = (torch.rand(N, 1, device=device) * 0.2 - 0.1).float()  # -0.1 到 0.1 m/s
        y = theoretical_stokes_v(vx).detach()
        return {'vx': vx, 'y': y}
    
    elif module_name == 'stokes_t':
        freq = torch.rand(N, 1, device=device) * 15000 + 5000
        
        t_list = []
        for i in range(N):
            period = 1.0 / freq[i].item()
            t_i = (torch.rand(1, device=device) * period).item()
            t_list.append(t_i)
        t = torch.tensor(t_list, device=device).unsqueeze(1).float()
        
        y = theoretical_stokes_t(t, freq).detach()
        return {'t': t, 'freq': freq, 'y': y}


def train_module(model, data, normalizer, module_name, device, target_loss=1e-5):
    """训练单个模块"""
    print(f"\n{'='*60}")
    print(f"训练模块: {module_name}")
    print(f"{'='*60}")
    
    # 归一化
    normalizer.fit(data)
    data_norm = normalizer.transform(data)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000)
    
    # 训练（无epoch上限，直到达到目标损失）
    print_interval = 2000
    losses = []
    epoch = 0
    
    while True:
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        if module_name == 'arf_x':
            pred = model(data_norm['x'], data_norm['SPL'], data_norm['freq'])
        elif module_name == 'arf_t':
            pred = model(data_norm['t'], data_norm['freq'])
        elif module_name == 'stokes_x':
            pred = model(data_norm['x'], data_norm['SPL'], data_norm['freq'])
        elif module_name == 'stokes_v':
            pred = model(data_norm['vx'])
        elif module_name == 'stokes_t':
            pred = model(data_norm['t'], data_norm['freq'])
        
        loss = criterion(pred, data_norm['y'])
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())
        
        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        
        if loss.item() < target_loss:
            print(f"  ✅ 达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        
        epoch += 1
    
    # 测试
    model.eval()
    with torch.no_grad():
        if module_name == 'arf_x':
            pred_norm = model(data_norm['x'], data_norm['SPL'], data_norm['freq'])
        elif module_name == 'arf_t':
            pred_norm = model(data_norm['t'], data_norm['freq'])
        elif module_name == 'stokes_x':
            pred_norm = model(data_norm['x'], data_norm['SPL'], data_norm['freq'])
        elif module_name == 'stokes_v':
            pred_norm = model(data_norm['vx'])
        elif module_name == 'stokes_t':
            pred_norm = model(data_norm['t'], data_norm['freq'])
        
        pred = normalizer.inverse_transform({'y': pred_norm})['y']
        true = data['y']
        
        rel_err = torch.abs(pred - true) / (torch.abs(true) + 1e-30)
        print(f"  平均相对误差: {rel_err.mean().item()*100:.2f}%")
        print(f"  最大相对误差: {rel_err.max().item()*100:.2f}%")
    
    return losses


def generate_3d_parameter_plots(combined_model, normalizers, save_dir, device):
    """生成3D参数影响图：仅绘制SPL vs 频率"""
    from mpl_toolkits.mplot3d import Axes3D
    
    combined_model.eval()
    
    # 固定条件
    x_fixed = 0.0
    t_fixed = 1e-5
    vx_fixed = 0.0
    
    resolution = 50  # 提高分辨率以获得更平滑的图像
    
    # 图: 频率 vs 声压级
    print("\n生成 3D 图: 频率 vs 声压级对受力的影响")
    freq_range = np.linspace(5000, 20000, resolution)
    SPL_range = np.linspace(120, 160, resolution)
    
    FREQ, SPL = np.meshgrid(freq_range, SPL_range)
    FX = np.zeros_like(FREQ)
    
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                x = torch.tensor([[x_fixed]], device=device, dtype=torch.float32)
                t = torch.tensor([[t_fixed]], device=device, dtype=torch.float32)
                vx = torch.tensor([[vx_fixed]], device=device, dtype=torch.float32)
                spl = torch.tensor([[SPL[i, j]]], device=device, dtype=torch.float32)
                freq = torch.tensor([[FREQ[i, j]]], device=device, dtype=torch.float32)
                
                # 归一化各个输入
                x_norm = normalizers['arf_x'].transform({'x': x})['x']
                t_norm = normalizers['arf_t'].transform({'t': t})['t']
                vx_norm = normalizers['stokes_v'].transform({'vx': vx})['vx']
                spl_norm = normalizers['arf_x'].transform({'SPL': spl})['SPL']
                freq_norm_arf_x = normalizers['arf_x'].transform({'freq': freq})['freq']
                freq_norm_arf_t = normalizers['arf_t'].transform({'freq': freq})['freq']
                freq_norm_stokes_x = normalizers['stokes_x'].transform({'freq': freq})['freq']
                freq_norm_stokes_t = normalizers['stokes_t'].transform({'freq': freq})['freq']
                
                # ARF
                f_arf_x = combined_model.arf_x(x_norm, spl_norm, freq_norm_arf_x)
                f_arf_t = combined_model.arf_t(t_norm, freq_norm_arf_t)
                f_arf_x = normalizers['arf_x'].inverse_transform({'y': f_arf_x})['y']
                f_arf_t = normalizers['arf_t'].inverse_transform({'y': f_arf_t})['y']
                F_arf = f_arf_x * f_arf_t
                
                # Stokes
                f_stokes_x = combined_model.stokes_x(x_norm, spl_norm, freq_norm_stokes_x)
                f_stokes_v = combined_model.stokes_v(vx_norm)
                f_stokes_t = combined_model.stokes_t(t_norm, freq_norm_stokes_t)
                f_stokes_x = normalizers['stokes_x'].inverse_transform({'y': f_stokes_x})['y']
                f_stokes_v = normalizers['stokes_v'].inverse_transform({'y': f_stokes_v})['y']
                f_stokes_t = normalizers['stokes_t'].inverse_transform({'y': f_stokes_t})['y']
                F_stokes = f_stokes_x * f_stokes_v * f_stokes_t
                
                fx = (F_arf + F_stokes).item() * 1e12
                FX[i, j] = fx
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(FREQ/1000, SPL, FX, cmap='viridis', alpha=0.9, edgecolor='none')
    ax.set_xlabel('频率 f (kHz)', fontsize=14, labelpad=10)
    ax.set_ylabel('声压级 SPL (dB)', fontsize=14, labelpad=10)
    ax.set_zlabel('受力 Fx (pN)', fontsize=14, labelpad=10)
    ax.set_title('频率与声压级对粒子受力的影响', fontsize=16, pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(save_dir / '3D_freq_vs_SPL.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存至: {save_dir / '3D_freq_vs_SPL.png'}")
    
    print("\n✅ 3D参数影响图已生成！")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*60)
    print("模块化PINN训练：分别训练五个子模块后组合")
    print("="*60)
    print(f"设备: {device}\n")
    
    save_dir = Path('PINN_wide')
    save_dir.mkdir(exist_ok=True)
    
    N_train = 50000  # 增加训练样本数以提高精度
    target_loss = 1e-4  # 降低目标损失以提高精度
    
    # 初始化模型
    arf_x_model = ARFNetX().to(device)
    arf_t_model = ARFNetT().to(device)
    stokes_x_model = StokesNetX().to(device)
    stokes_v_model = StokesNetV().to(device)
    stokes_t_model = StokesNetT().to(device)
    
    # 归一化器字典
    normalizers = {
        'arf_x': Normalizer(),
        'arf_t': Normalizer(),
        'stokes_x': Normalizer(),
        'stokes_v': Normalizer(),
        'stokes_t': Normalizer()
    }
    
    # 训练各个模块
    print("\n" + "="*60)
    print("第一阶段：分别训练五个子模块")
    print("="*60)
    
    # 1. 训练 ARF_X
    data_arf_x = generate_data_for_module('arf_x', N_train, device)
    train_module(arf_x_model, data_arf_x, normalizers['arf_x'], 'arf_x', device, target_loss)
    
    # 2. 训练 ARF_T
    data_arf_t = generate_data_for_module('arf_t', N_train, device)
    train_module(arf_t_model, data_arf_t, normalizers['arf_t'], 'arf_t', device, target_loss)
    
    # 3. 训练 Stokes_X
    data_stokes_x = generate_data_for_module('stokes_x', N_train, device)
    train_module(stokes_x_model, data_stokes_x, normalizers['stokes_x'], 'stokes_x', device, target_loss)
    
    # 4. 训练 Stokes_V
    data_stokes_v = generate_data_for_module('stokes_v', N_train, device)
    train_module(stokes_v_model, data_stokes_v, normalizers['stokes_v'], 'stokes_v', device, target_loss)
    
    # 5. 训练 Stokes_T
    data_stokes_t = generate_data_for_module('stokes_t', N_train, device)
    train_module(stokes_t_model, data_stokes_t, normalizers['stokes_t'], 'stokes_t', device, target_loss)
    
    # 组合模型
    print("\n" + "="*60)
    print("第二阶段：组合模型")
    print("="*60)
    
    combined_model = CombinedForceModel(
        arf_x_model, arf_t_model,
        stokes_x_model, stokes_v_model, stokes_t_model
    ).to(device)
    
    print("✅ 模型组合完成！")
    print(f"  公式: F_total = F_ARF + F_Stokes")
    print(f"  其中: F_ARF = f_ARF_x(x,SPL,freq) × f_ARF_t(t,freq)")
    print(f"       F_Stokes = f_Stokes_x(x,SPL,freq) × f_Stokes_v(vx) × f_Stokes_t(t,freq)")
    
    # 保存所有模型
    print("\n" + "="*60)
    print("保存模型...")
    print("="*60)
    
    torch.save(arf_x_model.state_dict(), save_dir / 'arf_x_model.pth')
    torch.save(arf_t_model.state_dict(), save_dir / 'arf_t_model.pth')
    torch.save(stokes_x_model.state_dict(), save_dir / 'stokes_x_model.pth')
    torch.save(stokes_v_model.state_dict(), save_dir / 'stokes_v_model.pth')
    torch.save(stokes_t_model.state_dict(), save_dir / 'stokes_t_model.pth')
    
    normalizers['arf_x'].save(save_dir / 'arf_x_norm.json')
    normalizers['arf_t'].save(save_dir / 'arf_t_norm.json')
    normalizers['stokes_x'].save(save_dir / 'stokes_x_norm.json')
    normalizers['stokes_v'].save(save_dir / 'stokes_v_norm.json')
    normalizers['stokes_t'].save(save_dir / 'stokes_t_norm.json')
    
    print("✅ 所有模块已保存")
    
    # 生成3D可视化
    print("\n" + "="*60)
    print("生成3D参数影响图...")
    print("="*60)
    generate_3d_parameter_plots(combined_model, normalizers, save_dir, device)
    
    # 保存配置
    config = {
        'model_type': 'Modular Combined Force Model (Enhanced)',
        'modules': ['arf_x', 'arf_t', 'stokes_x', 'stokes_v', 'stokes_t'],
        'input_params': ['x', 't', 'vx', 'SPL', 'freq'],
        'output_params': ['fx'],
        'training_samples_per_module': N_train,
        'target_loss': target_loss,
        'parameter_ranges': {
            'SPL': [120, 160],
            'freq': [5000, 20000]
        },
        'physical_constants': {
            'particle_radius': particle_radius,
            'particle_density': particle_density,
            't_relaxation': float(t_relaxation)
        },
        'improvements': [
            '提高了网络深度和宽度',
            '增加了训练样本数到50000',
            '降低了目标损失到1e-6',
            '增加了傅里叶特征数量'
        ]
    }
    
    with open(save_dir / 'model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print("="*60)
    print("\n模型文件:")
    print(f"  - arf_x_model.pth + arf_x_norm.json")
    print(f"  - arf_t_model.pth + arf_t_norm.json")
    print(f"  - stokes_x_model.pth + stokes_x_norm.json")
    print(f"  - stokes_v_model.pth + stokes_v_norm.json")
    print(f"  - stokes_t_model.pth + stokes_t_norm.json")
    print(f"  - model_config.json")
    print("\n3D可视化图:")
    print(f"  - 3D_freq_vs_SPL.png")
    print("\n改进:")
    print(f"  - 提高了网络深度和宽度")
    print(f"  - 训练样本数: {N_train}")
    print(f"  - 目标损失: {target_loss:.1e}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()


