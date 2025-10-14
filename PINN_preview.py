import numpy as np
import torch
import torch.nn as nn
import json
import os

# 导入初始化函数
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level

# 导入各个PINN模型
from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as ARFNormalizerX
from mechanisms.ARF_PINN_t import ARFNetT, Normalizer as ARFNormalizerT
from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV

# 导入原始公式
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf
from mechanisms.Stokes_drag import apply_stokes_drag

def load_individual_models():
    """加载五个单独的PINN模型"""
    print("正在加载五个单独的PINN模型...")
    
    models = {}
    normalizers = {}
    
    # 模型路径
    model_paths = {
        'arf_x': 'PINN/arf_model_x.pth',
        'arf_t': 'PINN/arf_model_t.pth',
        'stokes_x': 'PINN/stokes_model_x.pth',
        'stokes_t': 'PINN/stokes_model_t.pth',
        'stokes_v': 'PINN/stokes_model_v.pth'
    }
    
    norm_paths = {
        'arf_x': 'PINN/arf_model_x_normalization_params.json',
        'arf_t': 'PINN/arf_model_t_normalization_params.json',
        'stokes_x': 'PINN/stokes_model_x_normalization_params.json',
        'stokes_t': 'PINN/stokes_model_t_normalization_params.json',
        'stokes_v': 'PINN/stokes_model_v_normalization_params.json'
    }
    
    # 加载ARF x模型
    if os.path.exists(model_paths['arf_x']) and os.path.exists(norm_paths['arf_x']):
        models['arf_x'] = ARFNetX(fourier_features=32)
        models['arf_x'].load_state_dict(torch.load(model_paths['arf_x'], map_location='cpu', weights_only=True))
        models['arf_x'].eval()
        
        normalizers['arf_x'] = ARFNormalizerX()
        normalizers['arf_x'].load(norm_paths['arf_x'], device='cpu')
        print(f"成功加载ARF x模型: {model_paths['arf_x']}")
    else:
        print(f"ARF x模型文件不存在")
        return None, None
    
    # 加载ARF t模型
    if os.path.exists(model_paths['arf_t']) and os.path.exists(norm_paths['arf_t']):
        models['arf_t'] = ARFNetT(period_seconds=1.0/frequency)
        models['arf_t'].load_state_dict(torch.load(model_paths['arf_t'], map_location='cpu', weights_only=True))
        models['arf_t'].eval()
        
        normalizers['arf_t'] = ARFNormalizerT()
        normalizers['arf_t'].load(norm_paths['arf_t'], device='cpu')
        print(f"成功加载ARF t模型: {model_paths['arf_t']}")
    else:
        print(f"ARF t模型文件不存在")
        return None, None
    
    # 加载Stokes x模型
    if os.path.exists(model_paths['stokes_x']) and os.path.exists(norm_paths['stokes_x']):
        models['stokes_x'] = StokesNetX(fourier_features=32)
        models['stokes_x'].load_state_dict(torch.load(model_paths['stokes_x'], map_location='cpu', weights_only=True))
        models['stokes_x'].eval()
        
        normalizers['stokes_x'] = StokesNormalizerX()
        normalizers['stokes_x'].load(norm_paths['stokes_x'], device='cpu')
        print(f"成功加载Stokes x模型: {model_paths['stokes_x']}")
    else:
        print(f"Stokes x模型文件不存在")
        return None, None
    
    # 加载Stokes t模型
    if os.path.exists(model_paths['stokes_t']) and os.path.exists(norm_paths['stokes_t']):
        models['stokes_t'] = StokesNetT(period_seconds=1.0/frequency)
        models['stokes_t'].load_state_dict(torch.load(model_paths['stokes_t'], map_location='cpu', weights_only=True))
        models['stokes_t'].eval()
        
        normalizers['stokes_t'] = StokesNormalizerT()
        normalizers['stokes_t'].load(norm_paths['stokes_t'], device='cpu')
        print(f"成功加载Stokes t模型: {model_paths['stokes_t']}")
    else:
        print(f"Stokes t模型文件不存在")
        return None, None
    
    # 加载Stokes v模型
    if os.path.exists(model_paths['stokes_v']) and os.path.exists(norm_paths['stokes_v']):
        models['stokes_v'] = StokesNetV()
        models['stokes_v'].load_state_dict(torch.load(model_paths['stokes_v'], map_location='cpu', weights_only=True))
        models['stokes_v'].eval()
        
        normalizers['stokes_v'] = StokesNormalizerV()
        normalizers['stokes_v'].load(norm_paths['stokes_v'], device='cpu')
        print(f"成功加载Stokes v模型: {model_paths['stokes_v']}")
    else:
        print(f"Stokes v模型文件不存在")
        return None, None
    
    return models, normalizers

def compute_pinn_forces(positions, velocities, time, models, normalizers):
    """使用五个PINN模型计算力"""
    if models is None or normalizers is None:
        return None
    
    try:
        with torch.no_grad():
            # 转换为torch张量
            positions_t = torch.tensor(positions, dtype=torch.float32)
            velocities_t = torch.tensor(velocities, dtype=torch.float32)
            
            x = positions_t[:, 0]
            vx = velocities_t[:, 0]
            t_vec = torch.full_like(x, float(time))
            
            # 归一化输入 - 使用正确的归一化方法
            x_norm_arf = normalizers['arf_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
            t_norm_arf = normalizers['arf_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
            x_norm_stokes = normalizers['stokes_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
            t_norm_stokes = normalizers['stokes_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
            vx_norm_stokes = normalizers['stokes_v'].transform({'vx': vx.unsqueeze(1)})['vx'].squeeze()
            
            # 计算各个分量
            arf_fx = models['arf_x'](x_norm_arf.unsqueeze(1))
            arf_ft = models['arf_t'](t_norm_arf.unsqueeze(1))
            stokes_fx_raw = models['stokes_x'](x_norm_stokes.unsqueeze(1))
            stokes_ft_raw = models['stokes_t'](t_norm_stokes.unsqueeze(1))
            stokes_fv_raw = models['stokes_v'](vx_norm_stokes.unsqueeze(1))
            
            # 反归一化 ARF
            arf_fx = normalizers['arf_x'].inverse({'fx': arf_fx})['fx'].squeeze()
            arf_ft = normalizers['arf_t'].inverse({'fx': arf_ft})['fx'].squeeze()

            # 使用现有Stokes模型作为“因子”预测器：压缩为[-1,1]
            stokes_fx_factor = torch.tanh(stokes_fx_raw.squeeze())
            stokes_ft_factor = torch.tanh(stokes_ft_raw.squeeze())
            # 速度项采用真实 vx（更稳健）
            vx_real = vx
            
            # 组合力
            arf_total = arf_fx * arf_ft

            # 物理系数
            drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
            sound_pressure = 20e-6 * (10 ** (140 / 20))
            A = sound_pressure / (1.225 * 340 * 2 * np.pi * 10000)
            stokes_coeff = drag_coeff * 2 * np.pi * 10000 * A

            # 使用现有模型的因子重构Stokes力（与 ARF_STOKES/ Stokes_drag 保持一致相位：cos×cos）
            stokes_total = -drag_coeff * vx_real - stokes_coeff * stokes_fx_factor * stokes_ft_factor

            total_fx = arf_total + stokes_total
            
            return {
                'total_fx': total_fx,
                'arf_fx': arf_total,
                'stokes_fx': stokes_total
            }
            
    except Exception as e:
        print(f"PINN模型计算力时出错: {e}")
        return None

def compute_theoretical_forces(positions, velocities, radii, time):
    """使用原始公式计算理论力"""
    # 计算ARF力（保持不变）
    arf_force = compute_pressure_gradient_and_apply_arf(positions, radii, time)

    # 计算Stokes力（采用 cos 相位约定）：
    # u(x,t) = -2πfA cos(kx) cos(ωt)
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    N = positions.shape[0]

    sound_speed = 340.0
    wavelength = sound_speed / frequency
    k = 2.0 * np.pi / wavelength
    omega = 2.0 * np.pi * frequency

    reference_pressure = 20e-6
    rho_0 = 1.225
    c_0 = 340.0
    sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
    A = sound_pressure / (rho_0 * c_0 * 2.0 * np.pi * frequency)

    x_positions = positions[:, 0]
    u_x = -2.0 * np.pi * frequency * A * np.cos(k * x_positions) * np.cos(omega * time)

    drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
    relative_vx = velocities[:, 0] - u_x
    stokes_fx = -drag_coeff * relative_vx
    stokes_force = np.zeros_like(positions)
    stokes_force[:, 0] = stokes_fx

    # 总力
    total_force = arf_force + stokes_force

    return {
        'total_fx': total_force[:, 0],
        'arf_fx': arf_force[:, 0],
        'stokes_fx': stokes_force[:, 0]
    }

def main():
    """主函数"""
    print("=== PINN模型力计算预览 ===")
    
    # 加载模型
    models, normalizers = load_individual_models()
    if models is None or normalizers is None:
        print("模型加载失败")
        return
    
    # 初始化粒子
    domain_size = (0.034, 0.034)  # 34mm x 34mm
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=1000, domain_size=domain_size)
    
    print(f"\n初始化了 {len(positions_np)} 个粒子")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
    # 计算t=0时的力
    time = 0.0
    print(f"\n=== 时间 t = {time:.1e} s 的力计算 ===")
    
    # 使用PINN模型计算
    pinn_forces = compute_pinn_forces(positions_np, velocities_np, time, models, normalizers)
    
    # 使用原始公式计算
    theoretical_forces = compute_theoretical_forces(positions_np, velocities_np, radii_np, time)
    
    if pinn_forces is not None:
        print(f"\n--- PINN模型结果 ---")
        print(f"总力范围: [{pinn_forces['total_fx'].min():.2e}, {pinn_forces['total_fx'].max():.2e}] N")
        print(f"总力均值: {pinn_forces['total_fx'].mean():.2e} N")
        print(f"ARF力范围: [{pinn_forces['arf_fx'].min():.2e}, {pinn_forces['arf_fx'].max():.2e}] N")
        print(f"ARF力均值: {pinn_forces['arf_fx'].mean():.2e} N")
        print(f"Stokes力范围: [{pinn_forces['stokes_fx'].min():.2e}, {pinn_forces['stokes_fx'].max():.2e}] N")
        print(f"Stokes力均值: {pinn_forces['stokes_fx'].mean():.2e} N")
    
    print(f"\n--- 原始公式结果 ---")
    print(f"总力范围: [{theoretical_forces['total_fx'].min():.2e}, {theoretical_forces['total_fx'].max():.2e}] N")
    print(f"总力均值: {theoretical_forces['total_fx'].mean():.2e} N")
    print(f"ARF力范围: [{theoretical_forces['arf_fx'].min():.2e}, {theoretical_forces['arf_fx'].max():.2e}] N")
    print(f"ARF力均值: {theoretical_forces['arf_fx'].mean():.2e} N")
    print(f"Stokes力范围: [{theoretical_forces['stokes_fx'].min():.2e}, {theoretical_forces['stokes_fx'].max():.2e}] N")
    print(f"Stokes力均值: {theoretical_forces['stokes_fx'].mean():.2e} N")
    
    # 比较结果
    if pinn_forces is not None:
        print(f"\n--- 结果比较 ---")
        total_diff = np.abs(pinn_forces['total_fx'] - theoretical_forces['total_fx'])
        arf_diff = np.abs(pinn_forces['arf_fx'] - theoretical_forces['arf_fx'])
        stokes_diff = np.abs(pinn_forces['stokes_fx'] - theoretical_forces['stokes_fx'])
        
        print(f"总力平均差异: {total_diff.mean():.2e} N")
        print(f"ARF力平均差异: {arf_diff.mean():.2e} N")
        print(f"Stokes力平均差异: {stokes_diff.mean():.2e} N")
        
        # 计算相对误差
        total_rel_error = (total_diff / (np.abs(theoretical_forces['total_fx']) + 1e-30)).mean() * 100
        arf_rel_error = (arf_diff / (np.abs(theoretical_forces['arf_fx']) + 1e-30)).mean() * 100
        stokes_rel_error = (stokes_diff / (np.abs(theoretical_forces['stokes_fx']) + 1e-30)).mean() * 100
        
        print(f"总力相对误差: {total_rel_error:.2f}%")
        print(f"ARF力相对误差: {arf_rel_error:.2f}%")
        print(f"Stokes力相对误差: {stokes_rel_error:.2f}%")
    
    print("\n预览完成！")

if __name__ == "__main__":
    main()