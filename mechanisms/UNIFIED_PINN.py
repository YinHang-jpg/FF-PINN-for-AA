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
       
        # 加载Stokes x模型归一化参数（包含 x 范围）
        if os.path.exists(stokes_x_norm_path):
            with open(stokes_x_norm_path, 'r') as f:
                params = json.load(f)
            # Stokes 专用 x 范围
            self.stats['stokes_x_min'] = params.get('x_min', self.stats.get('x_min', 0.0))
            self.stats['stokes_x_max'] = params.get('x_max', self.stats.get('x_max', 0.1))
            self.stats['stokes_fx_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_fx_sigma'] = params.get('force_sigma', 1.0)
       
        # 加载Stokes t模型归一化参数（包含 t 范围）
        if os.path.exists(stokes_t_norm_path):
            with open(stokes_t_norm_path, 'r') as f:
                params = json.load(f)
            # Stokes 专用 t 范围
            self.stats['stokes_t_min'] = params.get('t_min', self.stats.get('t_min', 0.0))
            self.stats['stokes_t_max'] = params.get('t_max', self.stats.get('t_max', 1.0/frequency))
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
        对于时间变量，使用周期性归一化以支持多周期
        """
        # 位置归一化：对 x 做周期性映射，避免超出训练范围后被夹死在边界
        x_min = self.stats['x_min']
        x_max = self.stats['x_max']
        x_range = x_max - x_min
        if x_range <= 0:
            raise ValueError(f"Invalid x_range in UnifiedNormalizer: x_min={x_min}, x_max={x_max}")
        # 将任意 x 映射到 [x_min, x_max) 区间内（周期性扩展训练范围）
        x_wrapped = ((x - x_min) % x_range) + x_min
        x_norm = (x_wrapped - x_min) / x_range
       
        # 速度归一化
        vx_norm = (vx - self.stats['vx_min']) / (self.stats['vx_max'] - self.stats['vx_min'])
        vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
       
        # 时间归一化（周期性处理）
        period = self.stats['t_max'] - self.stats['t_min']  # 训练时的时间周期
        t_norm = ((t - self.stats['t_min']) % period) / period
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
       
        return x_norm, vx_norm, t_norm
 
    def normalize_inputs_stokes(self, x, vx, t):
        """
        使用 Stokes 自己的范围进行归一化（避免与 ARF 范围混淆）
        对于时间变量，使用周期性归一化以支持多周期
        """
        # 位置归一化（Stokes x 范围）
        x_min = self.stats.get('stokes_x_min', self.stats['x_min'])
        x_max = self.stats.get('stokes_x_max', self.stats['x_max'])
        x_range = x_max - x_min
        if x_range <= 0:
            raise ValueError(f"Invalid stokes_x_range in UnifiedNormalizer: x_min={x_min}, x_max={x_max}")
        # 同样对 Stokes 的 x 做周期性映射
        x_wrapped = ((x - x_min) % x_range) + x_min
        x_norm = (x_wrapped - x_min) / x_range
 
        # 速度归一化（沿用 vx 范围）
        vx_norm = (vx - self.stats['vx_min']) / (self.stats['vx_max'] - self.stats['vx_min'])
        vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
 
        # 时间归一化（周期性处理）
        t_min = self.stats.get('stokes_t_min', self.stats['t_min'])
        t_max = self.stats.get('stokes_t_max', self.stats['t_max'])
        period = t_max - t_min  # 训练时的时间周期
       
        # 将时间映射到 [0, 1] 范围内，支持周期性
        t_norm = ((t - t_min) % period) / period
        # 确保在 [0, 1] 范围内
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
            nn.Linear(fourier_features * 4, 128),  # 仅位置/时间 sin/cos 特征
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)  # 输出 ARF Fx
        )
       
        # Stokes分支网络
        self.stokes_net = nn.Sequential(
            nn.Linear(fourier_features * 4, 128),  # 仅位置/时间 sin/cos 特征
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
        # 使用标准的 sin/cos 特征，避免表达能力退化
        fourier_features_x = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
       
        # 生成时间相关的傅里叶特征
        fourier_t = self.fourier_weights_t * t
        fourier_features_t = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
       
        # 组合所有特征（仅 sin/cos 周期特征）
        combined_input = torch.cat([fourier_features_x, fourier_features_t], dim=1)
       
        # 分别预测ARF和Stokes力
        arf_fx = self.arf_net(combined_input)
        stokes_fx = self.stokes_net(combined_input)
       
        # 总力 = ARF + Stokes
        total_fx = arf_fx + stokes_fx
       
        return total_fx, arf_fx, stokes_fx
 
 
class PhysicalUnifiedModel(nn.Module):
    """运行时物理统一模型：
    - 内部处理输入归一化
    - 子模型推理
    - 输出在 forward 内完成反归一化与物理组合
    - forward 返回物理量纲的总力 Fx（N）
    """
    def __init__(self, models, unified_norm, device='cpu'):
        super().__init__()
        self.models = models
        self.unified_norm = unified_norm
        self.device = torch.device(device)
        
        # 从归一化参数中推断频率
        t_max = unified_norm.stats.get('t_max', 1.0/10000)
        self.inferred_frequency = 1.0 / t_max if t_max > 0 else 10000
        print(f"  [PhysicalUnifiedModel] 从归一化参数推断频率: {self.inferred_frequency:.1f} Hz (t_max={t_max:.6f}s)")
        print(f"  [PhysicalUnifiedModel] 对应波长: {340.0/self.inferred_frequency*1000:.2f} mm")
        print(f"  [PhysicalUnifiedModel] 驻波节点间距: {340.0/self.inferred_frequency*1000/2:.2f} mm")
 
    @torch.no_grad()
    def forward(self, x, vx, t):
        # x, vx, t: 1D Tensor (N,)
        x = x.to(self.device).float()
        vx = vx.to(self.device).float()
        t = t.to(self.device).float()
 
        # ARF 分量（声辐射力）
        x_norm, _, t_norm = self.unified_norm.normalize_inputs(x, vx, t)
        arf_fx_norm = self.models['arf_x'](x_norm.unsqueeze(1))
        arf_ft_norm = self.models['arf_t'](t_norm.unsqueeze(1))
        arf_fx, arf_ft = self.unified_norm.denormalize_arf_force(arf_fx_norm, arf_ft_norm)
        arf_total = arf_fx.squeeze() * arf_ft.squeeze()
 
        # Stokes 分量（各自范围归一化 → 模型 → 反归一化 → 物理组合）
        sx_norm, sv_norm, st_norm = self.unified_norm.normalize_inputs_stokes(x, vx, t)
        stokes_fx_norm = self.models['stokes_x'](sx_norm.unsqueeze(1))
        stokes_ft_norm = self.models['stokes_t'](st_norm.unsqueeze(1))
        stokes_fv_norm = self.models['stokes_v'](sv_norm.unsqueeze(1))
        stokes_fx, stokes_ft, stokes_fv = self.unified_norm.denormalize_stokes_force(
            stokes_fx_norm, stokes_ft_norm, stokes_fv_norm
        )
 
        # 物理常数与组合（修正斯托克斯阻力公式）：
        # F = -3*pi*1.86e-5*2e-6*(v - u_air)/1.0817
        # 其中 u_air = -2*pi*f*A*cos(2*pi*x/lambda)*cos(2*pi*f*t)
        mu = 1.86e-5
        cunningham = 1.0817
        diameter = 2e-6
        omega = 2.0 * np.pi * self.inferred_frequency
        drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
       
        # 动态计算振幅 A（根据频率和声压级）
        reference_pressure = 20e-6  # Pa
        sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
        rho_0 = 1.225  # 空气密度 (kg/m³)
        c_0 = 340      # 声速 (m/s)
        A = sound_pressure / (rho_0 * c_0 * omega)
       
        # 气流速度：u_air = -omega*A*cos(kx)*cos(omega*t)
        u_air = -omega * A * stokes_fx.squeeze() * stokes_ft.squeeze()
       
        # 斯托克斯阻力：F = -drag_coeff * (v - u_air)
        stokes_total = -drag_coeff * (vx - u_air)
 
        # 总力 = ARF + Stokes
        total_fx = arf_total + stokes_total
        return total_fx  # (N,)
 
def load_individual_models(device='cpu', model_base_path='PINN'):
    """
    加载各个单独的PINN模型
   
    Args:
        device: torch设备
        model_base_path: 模型文件所在的基础路径（例如 'PINN/freq_8k'）
    """
    models = {}
    
    # 从路径推断频率（用于创建模型实例）
    import re
    freq_match = re.search(r'freq_(\d+)k', model_base_path)
    if freq_match:
        inferred_frequency = int(freq_match.group(1)) * 1000  # 转换为Hz
        print(f"  从路径推断频率: {inferred_frequency} Hz")
    else:
        inferred_frequency = frequency # Default fallback
        print(f"  使用默认频率: {inferred_frequency} Hz")
   
    # 加载ARF模型
    try:
        from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as ARFNormalizerX
        from mechanisms.ARF_PINN_t import ARFNetT, Normalizer as ARFNormalizerT
       
        # ARF x模型
        arf_x_model = ARFNetX(fourier_features=32).to(device)
        arf_x_path = os.path.join(model_base_path, 'arf_model_x.pth')
        if os.path.exists(arf_x_path):
            arf_x_model.load_state_dict(torch.load(arf_x_path, map_location=device))
            models['arf_x'] = arf_x_model
            print(f"  ✓ 加载 ARF x模型: {arf_x_path}")
        else:
            print(f"  ✗ 未找到 ARF x模型: {arf_x_path}")
       
        # ARF t模型 - 使用推断的频率
        period = 1.0 / inferred_frequency
        arf_t_model = ARFNetT(period_seconds=period).to(device)
        arf_t_path = os.path.join(model_base_path, 'arf_model_t.pth')
        if os.path.exists(arf_t_path):
            arf_t_model.load_state_dict(torch.load(arf_t_path, map_location=device))
            models['arf_t'] = arf_t_model
            print(f"  ✓ 加载 ARF t模型: {arf_t_path}")
        else:
            print(f"  ✗ 未找到 ARF t模型: {arf_t_path}")
       
        # ARF归一化器
        arf_x_normalizer = ARFNormalizerX()
        arf_x_norm_path = os.path.join(model_base_path, 'arf_model_x_normalization_params.json')
        if os.path.exists(arf_x_norm_path):
            arf_x_normalizer.load(arf_x_norm_path, device)
            models['arf_x_norm'] = arf_x_normalizer
            print(f"  ✓ 加载 ARF x归一化参数")
        else:
            print(f"  ✗ 未找到 ARF x归一化参数: {arf_x_norm_path}")
       
        arf_t_normalizer = ARFNormalizerT()
        arf_t_norm_path = os.path.join(model_base_path, 'arf_model_t_normalization_params.json')
        if os.path.exists(arf_t_norm_path):
            arf_t_normalizer.load(arf_t_norm_path, device)
            models['arf_t_norm'] = arf_t_normalizer
            print(f"  ✓ 加载 ARF t归一化参数")
        else:
            print(f"  ✗ 未找到 ARF t归一化参数: {arf_t_norm_path}")
           
    except Exception as e:
        print(f"加载ARF模型时出错: {e}")
   
    # 加载Stokes模型
    try:
        from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
        from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
        from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV
       
        # Stokes x模型
        stokes_x_model = StokesNetX(fourier_features=32).to(device)
        stokes_x_path = os.path.join(model_base_path, 'stokes_model_x.pth')
        if os.path.exists(stokes_x_path):
            stokes_x_model.load_state_dict(torch.load(stokes_x_path, map_location=device))
            models['stokes_x'] = stokes_x_model
            print(f"  ✓ 加载 Stokes x模型: {stokes_x_path}")
        else:
            print(f"  ✗ 未找到 Stokes x模型: {stokes_x_path}")
       
        # Stokes t模型 - 使用推断的频率
        period = 1.0 / inferred_frequency
        stokes_t_model = StokesNetT(period_seconds=period).to(device)
        stokes_t_path = os.path.join(model_base_path, 'stokes_model_t.pth')
        if os.path.exists(stokes_t_path):
            stokes_t_model.load_state_dict(torch.load(stokes_t_path, map_location=device))
            models['stokes_t'] = stokes_t_model
            print(f"  ✓ 加载 Stokes t模型: {stokes_t_path}")
        else:
            print(f"  ✗ 未找到 Stokes t模型: {stokes_t_path}")
       
        # Stokes v模型
        stokes_v_model = StokesNetV().to(device)
        stokes_v_path = os.path.join(model_base_path, 'stokes_model_v.pth')
        if os.path.exists(stokes_v_path):
            stokes_v_model.load_state_dict(torch.load(stokes_v_path, map_location=device))
            models['stokes_v'] = stokes_v_model
            print(f"  ✓ 加载 Stokes v模型: {stokes_v_path}")
        else:
            print(f"  ✗ 未找到 Stokes v模型: {stokes_v_path}")
       
        # Stokes归一化器
        stokes_x_normalizer = StokesNormalizerX()
        stokes_x_norm_path = os.path.join(model_base_path, 'stokes_model_x_normalization_params.json')
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
            print(f"  ✓ 加载 Stokes x归一化参数")
        else:
            print(f"  ✗ 未找到 Stokes x归一化参数: {stokes_x_norm_path}")
       
        stokes_t_normalizer = StokesNormalizerT()
        stokes_t_norm_path = os.path.join(model_base_path, 'stokes_model_t_normalization_params.json')
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
            print(f"  ✓ 加载 Stokes t归一化参数")
        else:
            print(f"  ✗ 未找到 Stokes t归一化参数: {stokes_t_norm_path}")
       
        stokes_v_normalizer = StokesNormalizerV()
        stokes_v_norm_path = os.path.join(model_base_path, 'stokes_model_v_normalization_params.json')
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
            print(f"  ✓ 加载 Stokes v归一化参数")
        else:
            print(f"  ✗ 未找到 Stokes v归一化参数: {stokes_v_norm_path}")
           
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
            # ARF力（声辐射力）
            arf_fx_norm = models['arf_x'](x_norm.unsqueeze(1))
            arf_ft_norm = models['arf_t'](t_norm.unsqueeze(1))
            arf_fx, arf_ft = models['unified_norm'].denormalize_arf_force(arf_fx_norm, arf_ft_norm)
            arf_total = arf_fx.squeeze() * arf_ft.squeeze()
           
            # Stokes力（模型输出反归一化，按 cos 相位组合）
            stokes_x_norm, stokes_v_norm, stokes_t_norm = models['unified_norm'].normalize_inputs_stokes(x, vx, t)
            stokes_fx_norm = models['stokes_x'](stokes_x_norm.unsqueeze(1))
            stokes_ft_norm = models['stokes_t'](stokes_t_norm.unsqueeze(1))
            stokes_fv_norm = models['stokes_v'](stokes_v_norm.unsqueeze(1))
            stokes_fx, stokes_ft, stokes_fv = models['unified_norm'].denormalize_stokes_force(
                stokes_fx_norm, stokes_ft_norm, stokes_fv_norm)
            # 物理常数与组合（修正斯托克斯阻力公式）：
            # F = -3*pi*1.86e-5*2e-6*(v - u_air)/1.0817
            # 其中 u_air = -2*pi*f*A*cos(2*pi*x/lambda)*cos(2*pi*f*t)
            mu = 1.86e-5
            cunningham = 1.0817
            diameter = 2e-6
            omega = 2.0 * np.pi * frequency
            drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
           
            # 动态计算振幅 A（根据频率和声压级）
            reference_pressure = 20e-6  # Pa
            sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
            rho_0 = 1.225  # 空气密度 (kg/m³)
            c_0 = 340      # 声速 (m/s)
            A = sound_pressure / (rho_0 * c_0 * omega)
           
            # 气流速度：u_air = -omega*A*cos(kx)*cos(omega*t)
            u_air = -omega * A * stokes_fx.squeeze() * stokes_ft.squeeze()
           
            # 斯托克斯阻力：F = -drag_coeff * (v - u_air)
            stokes_total = -drag_coeff * (vx - u_air)
           
            # 总力 = ARF + Stokes
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
    model_base_path = 'PINN'  # 默认路径
    models = load_individual_models(device, model_base_path)
   
    # 创建统一归一化器
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': os.path.join(model_base_path, 'arf_model_x_normalization_params.json'),
        'arf_t_norm': os.path.join(model_base_path, 'arf_model_t_normalization_params.json'),
        'stokes_x_norm': os.path.join(model_base_path, 'stokes_model_x_normalization_params.json'),
        'stokes_t_norm': os.path.join(model_base_path, 'stokes_model_t_normalization_params.json'),
        'stokes_v_norm': os.path.join(model_base_path, 'stokes_model_v_normalization_params.json')
    }
    unified_norm.load_from_individual_models(model_paths, device)
    models['unified_norm'] = unified_norm
 
    # 构建运行时物理模型（内部完成反归一化与物理组合）
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
   
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
   
    # 动画：时间 0 到 0.1 ms，步长 10 µs；四个子图，循环播放
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        print("\n生成合并模型动画(0~0.1ms, dt=10µs, 循环)...")
 
        # 网格定义
        x_line = torch.linspace(0.0, c_0 / frequency, 200, device=device)
        vx_line = torch.linspace(-0.1, 0.1, 200, device=device)
        pos_x = torch.stack([x_line, torch.zeros_like(x_line)], dim=1).cpu().numpy()
        pos_zero = np.zeros((vx_line.shape[0], 2), dtype=pos_x.dtype)
        vel_zero = np.zeros((pos_x.shape[0], 2), dtype=pos_x.dtype)
        vel_v = torch.stack([vx_line, torch.zeros_like(vx_line)], dim=1).cpu().numpy()
        # 随机粒子（演示合力散点）
        rng = np.random.default_rng(42)
        Np = 300
        part_pos = np.zeros((Np, 2), dtype=pos_x.dtype)
        part_pos[:, 0] = rng.uniform(0.0, float(c_0 / frequency), size=Np)
        part_vel = np.zeros_like(part_pos)
 
        # 动画时间轴（秒）
        t_frames = np.arange(0.0, 1e-4 + 1e-12, 1e-5)  # 0..0.1ms，步长10µs
       
        # 为绘制平滑曲线创建更密集的时间采样点
        t_plot = np.linspace(0.0, 1e-4, 200)  # 200个点用于平滑绘图
 
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        ax_x, ax_v, ax_t, ax_p = axes.flatten()
 
        # 初始化图元
        line_x, = ax_x.plot((x_line * 1000).cpu().numpy(), np.zeros_like(x_line.cpu().numpy()), 'b-')
        ax_x.set_xlabel('x (mm)'); ax_x.set_ylabel('Fx (N)'); ax_x.set_title('Fx vs x (vx=0)'); ax_x.grid(True, alpha=0.3)
 
        line_v, = ax_v.plot(vx_line.cpu().numpy(), np.zeros_like(vx_line.cpu().numpy()), 'g-')
        ax_v.set_xlabel('vx (m/s)'); ax_v.set_ylabel('Fx (N)'); ax_v.set_title('Fx vs vx (x=0)'); ax_v.grid(True, alpha=0.3)
 
        t_us_plot = t_plot * 1e6  # 用于绘图的时间轴（µs）
        line_t, = ax_t.plot(t_us_plot, np.zeros_like(t_us_plot), 'r-')
        ax_t.set_xlabel('t (µs)'); ax_t.set_ylabel('Fx (N, mean)'); ax_t.set_title('Fx vs t (x=0, vx=0)'); ax_t.grid(True, alpha=0.3)
 
        scat_p = ax_p.scatter(part_pos[:, 0] * 1000.0, np.zeros_like(part_pos[:, 0]), s=1, c='purple', alpha=0.7)
        ax_p.set_xlabel('x (mm)'); ax_p.set_ylabel('Fx (N)'); ax_p.set_title('Particle Forces'); ax_p.grid(True, alpha=0.3)
 
        # 初始由数据自动缩放，后续在每帧更新时动态调整
 
        def update(frame_idx):
            tt = float(t_frames[frame_idx])
            # Fx(x)
            x_tensor = torch.tensor(pos_x[:, 0], device=device, dtype=torch.float32)
            vx_zero = torch.zeros_like(x_tensor)
            t_tensor = torch.full_like(x_tensor, tt)
            Fx_x = phys_model(x_tensor, vx_zero, t_tensor).cpu().numpy()
            line_x.set_ydata(Fx_x)
            # Fx(vx)
            vx_tensor = torch.tensor(vel_v[:, 0], device=device, dtype=torch.float32)
            x_zero = torch.zeros_like(vx_tensor)
            t_v = torch.full_like(vx_tensor, tt)
            Fx_v = phys_model(x_zero, vx_tensor, t_v).cpu().numpy()
            line_v.set_ydata(Fx_v)
            # Fx(t) 均值 - 使用密集采样点绘制平滑曲线
            Fx_t_vals = []
            for ttmp in t_plot:
                t_tmp_tensor = torch.full_like(x_zero, float(ttmp))
                Fx_tmp = phys_model(x_zero, vx_zero, t_tmp_tensor).cpu().numpy()
                Fx_t_vals.append(Fx_tmp.mean())
            line_t.set_ydata(np.array(Fx_t_vals))
            # 粒子合力散点
            xp = torch.tensor(part_pos[:, 0], device=device, dtype=torch.float32)
            vp = torch.tensor(part_vel[:, 0], device=device, dtype=torch.float32)
            tp = torch.full_like(xp, tt)
            Fx_p = phys_model(xp, vp, tp).cpu().numpy()
            scat_p.set_offsets(np.c_[part_pos[:, 0] * 1000.0, Fx_p])
            # 动态自适应各子图y轴范围
            ax_x.relim(); ax_x.autoscale_view()
            ax_v.relim(); ax_v.autoscale_view()
            ax_t.relim(); ax_t.autoscale_view()
            # 对散点图单独计算y范围（relim对PathCollection可能不生效）
            y_min = np.min(Fx_p)
            y_max = np.max(Fx_p)
            if y_min == y_max:
                y_pad = 1e-12 if y_min == 0 else abs(y_min) * 0.05
                y_min -= y_pad
                y_max += y_pad
            else:
                pad = (y_max - y_min) * 0.05
                y_min -= pad
                y_max += pad
            ax_p.set_ylim(y_min, y_max)
            return line_x, line_v, line_t, scat_p
 
        anim = FuncAnimation(fig, update, frames=len(t_frames), interval=200, blit=False, repeat=True)
        plt.show()
        print("动画生成完成（循环播放）。")
    except Exception as _e:
        print(f"生成动画失败: {_e}")
   
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