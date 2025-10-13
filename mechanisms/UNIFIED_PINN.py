import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os

# 导入物理参数
from initialization.sound_source_standing import frequency, sound_pressure_level, spl_to_pressure
from mechanisms.ARF import p_0, rho_0, gamma, c_0, get_amplitude
from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound

# 物理参数
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa


class UnifiedNormalizer:
    """
    统一归一化器，处理所有输入输出变量的归一化
    """
    def __init__(self):
        self.stats = {}

    def load_from_individual_models(self, model_paths, device='cpu'):
        """
        从各个单独模型的归一化参数文件中加载参数
        """
        # ARF模型参数
        arf_x_norm_path = os.path.join(model_paths['arf_x_norm'])
        arf_t_norm_path = os.path.join(model_paths['arf_t_norm'])
        
        # Stokes模型参数
        stokes_x_norm_path = os.path.join(model_paths['stokes_x_norm'])
        stokes_t_norm_path = os.path.join(model_paths['stokes_t_norm'])
        stokes_v_norm_path = os.path.join(model_paths['stokes_v_norm'])
        
        # 加载ARF x模型归一化参数
        if os.path.exists(arf_x_norm_path):
            with open(arf_x_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['x_min'] = params.get('x_min', 0.0)
            self.stats['x_max'] = params.get('x_max', 0.1)
            self.stats['arf_fx_mu'] = params.get('force_mu', 0.0)
            self.stats['arf_fx_sigma'] = params.get('force_sigma', 1.0)
        
        # 加载ARF t模型归一化参数
        if os.path.exists(arf_t_norm_path):
            with open(arf_t_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['t_min'] = params.get('t_min', 0.0)
            self.stats['t_max'] = params.get('t_max', 1.0/frequency)
            self.stats['arf_ft_mu'] = params.get('force_mu', 0.0)
            self.stats['arf_ft_sigma'] = params.get('force_sigma', 1.0)
        
        # 加载Stokes x模型归一化参数
        if os.path.exists(stokes_x_norm_path):
            with open(stokes_x_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['stokes_fx_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_fx_sigma'] = params.get('force_sigma', 1.0)
        
        # 加载Stokes t模型归一化参数
        if os.path.exists(stokes_t_norm_path):
            with open(stokes_t_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['stokes_ft_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_ft_sigma'] = params.get('force_sigma', 1.0)
        
        # 加载Stokes v模型归一化参数
        if os.path.exists(stokes_v_norm_path):
            with open(stokes_v_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['vx_min'] = params.get('vx_min', -0.1)
            self.stats['vx_max'] = params.get('vx_max', 0.1)
            self.stats['stokes_fv_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_fv_sigma'] = params.get('force_sigma', 1.0)

    def normalize_inputs(self, x, vx, t):
        """
        归一化输入变量
        """
        # 位置归一化
        x_norm = (x - self.stats['x_min']) / (self.stats['x_max'] - self.stats['x_min'])
        x_norm = torch.clamp(x_norm, 0.0, 1.0)
        
        # 速度归一化
        vx_norm = (vx - self.stats['vx_min']) / (self.stats['vx_max'] - self.stats['vx_min'])
        vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
        
        # 时间归一化
        t_norm = (t - self.stats['t_min']) / (self.stats['t_max'] - self.stats['t_min'])
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
        
        return x_norm, vx_norm, t_norm

    def denormalize_arf_force(self, fx_norm, ft_norm):
        """
        反归一化ARF力
        """
        fx = fx_norm * self.stats['arf_fx_sigma'] + self.stats['arf_fx_mu']
        ft = ft_norm * self.stats['arf_ft_sigma'] + self.stats['arf_ft_mu']
        return fx, ft

    def denormalize_stokes_force(self, fx_norm, ft_norm, fv_norm):
        """
        反归一化Stokes力
        """
        fx = fx_norm * self.stats['stokes_fx_sigma'] + self.stats['stokes_fx_mu']
        ft = ft_norm * self.stats['stokes_ft_sigma'] + self.stats['stokes_ft_mu']
        fv = fv_norm * self.stats['stokes_fv_sigma'] + self.stats['stokes_fv_mu']
        return fx, ft, fv


class UnifiedPINN(nn.Module):
    """
    统一的PINN模型，结合ARF和Stokes力预测
    输入: [x, vx, t] (位置x, 速度vx, 时间t)
    输出: [Fx] (x方向合力 = ARF + Stokes)
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算波数用于傅里叶特征
        wavelength = c_0 / frequency
        k = 2 * np.pi / wavelength
        omega = 2 * np.pi * frequency
        
        # 位置相关的傅里叶特征权重
        self.register_buffer('fourier_weights_x', torch.randn(1, fourier_features) * k)
        
        # 时间相关的傅里叶特征权重
        self.register_buffer('fourier_weights_t', torch.randn(1, fourier_features) * omega)
        
        # ARF分支网络
        self.arf_net = nn.Sequential(
            nn.Linear(fourier_features * 4 + 3, 128),  # 位置sin/cos + 时间sin/cos + x + vx + t
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # 输出 ARF Fx
        )
        
        # Stokes分支网络
        self.stokes_net = nn.Sequential(
            nn.Linear(fourier_features * 4 + 3, 128),  # 位置sin/cos + 时间sin/cos + x + vx + t
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # 输出 Stokes Fx
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
        
        # 分别预测ARF和Stokes力
        arf_fx = self.arf_net(combined_input)
        stokes_fx = self.stokes_net(combined_input)
        
        # 总力 = ARF + Stokes
        total_fx = arf_fx + stokes_fx
        
        return total_fx, arf_fx, stokes_fx


def load_individual_models(device='cpu'):
    """
    加载各个单独的PINN模型
    """
    models = {}
    
    # 加载ARF模型
    try:
        from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as ARFNormalizerX
        from mechanisms.ARF_PINN_t import ARFNetT, Normalizer as ARFNormalizerT
        
        # ARF x模型
        arf_x_model = ARFNetX(fourier_features=32).to(device)
        arf_x_path = 'PINN/arf_model_x.pth'
        if os.path.exists(arf_x_path):
            arf_x_model.load_state_dict(torch.load(arf_x_path, map_location=device))
            models['arf_x'] = arf_x_model
        
        # ARF t模型
        period = 1.0 / frequency
        arf_t_model = ARFNetT(period_seconds=period).to(device)
        arf_t_path = 'PINN/arf_model_t.pth'
        if os.path.exists(arf_t_path):
            arf_t_model.load_state_dict(torch.load(arf_t_path, map_location=device))
            models['arf_t'] = arf_t_model
        
        # ARF归一化器
        arf_x_normalizer = ARFNormalizerX()
        arf_x_norm_path = 'PINN/arf_model_x_normalization_params.json'
        if os.path.exists(arf_x_norm_path):
            arf_x_normalizer.load(arf_x_norm_path, device)
            models['arf_x_norm'] = arf_x_normalizer
        
        arf_t_normalizer = ARFNormalizerT()
        arf_t_norm_path = 'PINN/arf_model_t_normalization_params.json'
        if os.path.exists(arf_t_norm_path):
            arf_t_normalizer.load(arf_t_norm_path, device)
            models['arf_t_norm'] = arf_t_normalizer
            
    except Exception as e:
        print(f"加载ARF模型时出错: {e}")
    
    # 加载Stokes模型
    try:
        from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
        from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
        from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV
        
        # Stokes x模型
        stokes_x_model = StokesNetX(fourier_features=32).to(device)
        stokes_x_path = 'PINN/stokes_model_x.pth'
        if os.path.exists(stokes_x_path):
            stokes_x_model.load_state_dict(torch.load(stokes_x_path, map_location=device))
            models['stokes_x'] = stokes_x_model
        
        # Stokes t模型
        period = 1.0 / frequency
        stokes_t_model = StokesNetT(period_seconds=period).to(device)
        stokes_t_path = 'PINN/stokes_model_t.pth'
        if os.path.exists(stokes_t_path):
            stokes_t_model.load_state_dict(torch.load(stokes_t_path, map_location=device))
            models['stokes_t'] = stokes_t_model
        
        # Stokes v模型
        stokes_v_model = StokesNetV().to(device)
        stokes_v_path = 'PINN/stokes_model_v.pth'
        if os.path.exists(stokes_v_path):
            stokes_v_model.load_state_dict(torch.load(stokes_v_path, map_location=device))
            models['stokes_v'] = stokes_v_model
        
        # Stokes归一化器
        stokes_x_normalizer = StokesNormalizerX()
        stokes_x_norm_path = 'PINN/stokes_model_x_normalization_params.json'
        if os.path.exists(stokes_x_norm_path):
            with open(stokes_x_norm_path, 'r') as f:
                params = json.load(f)
            stokes_x_normalizer.stats = {
                'x': (torch.tensor(params['x_min'], dtype=torch.float32), 
                      torch.tensor(params['x_max'] - params['x_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_x_norm'] = stokes_x_normalizer
        
        stokes_t_normalizer = StokesNormalizerT()
        stokes_t_norm_path = 'PINN/stokes_model_t_normalization_params.json'
        if os.path.exists(stokes_t_norm_path):
            with open(stokes_t_norm_path, 'r') as f:
                params = json.load(f)
            stokes_t_normalizer.stats = {
                't': (torch.tensor(params['t_min'], dtype=torch.float32), 
                      torch.tensor(params['t_max'] - params['t_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_t_norm'] = stokes_t_normalizer
        
        stokes_v_normalizer = StokesNormalizerV()
        stokes_v_norm_path = 'PINN/stokes_model_v_normalization_params.json'
        if os.path.exists(stokes_v_norm_path):
            with open(stokes_v_norm_path, 'r') as f:
                params = json.load(f)
            stokes_v_normalizer.stats = {
                'vx': (torch.tensor(params['vx_min'], dtype=torch.float32), 
                       torch.tensor(params['vx_max'] - params['vx_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_v_norm'] = stokes_v_normalizer
            
    except Exception as e:
        print(f"加载Stokes模型时出错: {e}")
    
    return models


def compute_unified_force(positions, velocities, time, models, device='cpu'):
    """
    使用统一模型计算粒子合力
    """
    if not all(key in models for key in ['arf_x', 'arf_t', 'stokes_x', 'stokes_t', 'stokes_v']):
        print("部分模型未加载，无法计算统一力")
        return np.zeros_like(positions)
    
    try:
        with torch.no_grad():
            x = torch.tensor(positions[:, 0], dtype=torch.float32, device=device)
            vx = torch.tensor(velocities[:, 0], dtype=torch.float32, device=device)
            t = torch.full_like(x, time, dtype=torch.float32)
            
            # 归一化输入
            x_norm, vx_norm, t_norm = models['unified_norm'].normalize_inputs(x, vx, t)
            
            # 使用各个单独模型预测
            # ARF力
            arf_fx_norm = models['arf_x'](x_norm.unsqueeze(1))
            arf_ft_norm = models['arf_t'](t_norm.unsqueeze(1))
            arf_fx, arf_ft = models['unified_norm'].denormalize_arf_force(arf_fx_norm, arf_ft_norm)
            arf_total = arf_fx.squeeze() * arf_ft.squeeze()  # ARF = F_x(x) * F_t(t)
            
            # Stokes力
            stokes_fx_norm = models['stokes_x'](x_norm.unsqueeze(1))
            stokes_ft_norm = models['stokes_t'](t_norm.unsqueeze(1))
            stokes_fv_norm = models['stokes_v'](vx_norm.unsqueeze(1))
            stokes_fx, stokes_ft, stokes_fv = models['unified_norm'].denormalize_stokes_force(
                stokes_fx_norm, stokes_ft_norm, stokes_fv_norm)
            stokes_total = stokes_fv.squeeze() - stokes_fx.squeeze() * stokes_ft.squeeze()  # F_v - F_x * F_t
            
            # 总力
            total_fx = arf_total + stokes_total
            total_fy = torch.zeros_like(total_fx)
            
            return torch.stack([total_fx, total_fy], dim=1).cpu().numpy()
            
    except Exception as e:
        print(f"统一模型计算合力时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros_like(positions)


def main():
    """
    主函数：创建统一模型并测试
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载各个单独模型
    print("加载各个单独模型...")
    models = load_individual_models(device)
    
    # 创建统一归一化器
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': 'PINN/arf_model_x_normalization_params.json',
        'arf_t_norm': 'PINN/arf_model_t_normalization_params.json',
        'stokes_x_norm': 'PINN/stokes_model_x_normalization_params.json',
        'stokes_t_norm': 'PINN/stokes_model_t_normalization_params.json',
        'stokes_v_norm': 'PINN/stokes_model_v_normalization_params.json'
    }
    unified_norm.load_from_individual_models(model_paths, device)
    models['unified_norm'] = unified_norm
    
    # 检查模型是否全部加载成功
    required_models = ['arf_x', 'arf_t', 'stokes_x', 'stokes_t', 'stokes_v']
    missing_models = [model for model in required_models if model not in models]
    if missing_models:
        print(f"缺少模型: {missing_models}")
        return
    
    print("所有模型加载成功！")
    
    # 测试统一模型
    print("\n测试统一模型...")
    
    # 创建测试数据
    N = 100
    x_test = torch.linspace(0.001, 0.05, N, device=device)  # 1mm to 50mm
    vx_test = torch.linspace(-0.05, 0.05, N, device=device)  # -5cm/s to 5cm/s
    t_test = torch.linspace(0, 1.0/frequency, N, device=device)  # 一个周期
    
    positions = torch.stack([x_test, torch.zeros_like(x_test)], dim=1).cpu().numpy()
    velocities = torch.stack([vx_test, torch.zeros_like(vx_test)], dim=1).cpu().numpy()
    time = 0.0
    
    # 计算合力
    forces = compute_unified_force(positions, velocities, time, models, device)
    
    print(f"计算了 {N} 个粒子的合力")
    print(f"力的范围: Fx = [{forces[:, 0].min():.2e}, {forces[:, 0].max():.2e}] N")
    print(f"力的均值: Fx = {forces[:, 0].mean():.2e} N")
    
    # 保存统一模型
    unified_model = UnifiedPINN(fourier_features=32).to(device)
    torch.save(unified_model.state_dict(), 'PINN/unified_model.pth')
    
    # 保存统一归一化参数
    unified_norm_params = {
        'x_min': unified_norm.stats['x_min'],
        'x_max': unified_norm.stats['x_max'],
        'vx_min': unified_norm.stats['vx_min'],
        'vx_max': unified_norm.stats['vx_max'],
        't_min': unified_norm.stats['t_min'],
        't_max': unified_norm.stats['t_max'],
        'arf_fx_mu': unified_norm.stats['arf_fx_mu'],
        'arf_fx_sigma': unified_norm.stats['arf_fx_sigma'],
        'arf_ft_mu': unified_norm.stats['arf_ft_mu'],
        'arf_ft_sigma': unified_norm.stats['arf_ft_sigma'],
        'stokes_fx_mu': unified_norm.stats['stokes_fx_mu'],
        'stokes_fx_sigma': unified_norm.stats['stokes_fx_sigma'],
        'stokes_ft_mu': unified_norm.stats['stokes_ft_mu'],
        'stokes_ft_sigma': unified_norm.stats['stokes_ft_sigma'],
        'stokes_fv_mu': unified_norm.stats['stokes_fv_mu'],
        'stokes_fv_sigma': unified_norm.stats['stokes_fv_sigma']
    }
    
    with open('PINN/unified_model_normalization_params.json', 'w') as f:
        json.dump(unified_norm_params, f, indent=2)
    
    print("\n统一模型和归一化参数已保存到 PINN/ 目录")


if __name__ == "__main__":
    main()
