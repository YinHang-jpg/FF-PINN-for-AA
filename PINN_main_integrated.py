import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import time
from scipy.interpolate import PchipInterpolator

# 导入初始化函数
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level, compute_sound_field

# 导入各个PINN模型与归一化器（参考 PINN_preview.py）
from mechanisms.UNIFIED_PINN import (
    load_individual_models,
    UnifiedNormalizer,
    PhysicalUnifiedModel,
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理参数
USE_STOKES_DRAG = True  # 是否使用Stokes阻力
USE_ARF = True  # 是否使用声辐射力
dt = 1e-7  # 时间步长 (s)
gravity = 9.81  # 重力加速度 (m/s²)
particle_density = 1000  # 粒子密度 (kg/m³)
air_density = 1.225  # 空气密度 (kg/m³)
viscosity = 1.8e-5  # 空气动力粘度 (Pa·s)
fixed_diameter = 2e-6  # 固定粒子直径 (m)
fixed_cunningham = 1.083  # 固定Cunningham修正因子


def load_physical_unified_model(device):
    models = load_individual_models(device)
    # 统一归一化器聚合各子模型的归一化范围/统计
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': 'PINN/arf_model_x_normalization_params.json',
        'arf_t_norm': 'PINN/arf_model_t_normalization_params.json',
        'stokes_x_norm': 'PINN/stokes_model_x_normalization_params.json',
        'stokes_t_norm': 'PINN/stokes_model_t_normalization_params.json',
        'stokes_v_norm': 'PINN/stokes_model_v_normalization_params.json',
    }
    unified_norm.load_from_individual_models(model_paths, device='cpu')
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
    return phys_model

def compute_force_using_trained_models(positions_t, velocities_t, t_scalar, models, norms):
    with torch.no_grad():
        # 在CPU上运行模型，避免显存波动
        x = positions_t[:, 0].detach().cpu()
        vx = velocities_t[:, 0].detach().cpu()
        t_vec = torch.full_like(x, float(t_scalar))

        # 归一化
        x_norm_arf = norms['arf_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_arf = norms['arf_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        x_norm_stokes = norms['stokes_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_stokes = norms['stokes_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        vx_norm_stokes = norms['stokes_v'].transform({'vx': vx.unsqueeze(1)})['vx'].squeeze()

        # 模型输出
        arf_fx_norm = models['arf_x'](x_norm_arf.unsqueeze(1))
        arf_ft_norm = models['arf_t'](t_norm_arf.unsqueeze(1))
        stokes_fx_norm = models['stokes_x'](x_norm_stokes.unsqueeze(1))
        stokes_ft_norm = models['stokes_t'](t_norm_stokes.unsqueeze(1))
        stokes_fv_norm = models['stokes_v'](vx_norm_stokes.unsqueeze(1))

        # 反归一化 ARF 分量
        arf_fx = norms['arf_x'].inverse({'fx': arf_fx_norm})['fx'].squeeze()
        arf_ft = norms['arf_t'].inverse({'fx': arf_ft_norm})['fx'].squeeze()

        # 反归一化 Stokes 三个分量为物理因子/量
        # - stokes_x_factor ≈ cos(kx)
        # - stokes_t_factor ≈ cos(ωt)
        # - stokes_v_value  ≈ vx
        stokes_fx_factor = norms['stokes_x'].inverse({'fx': stokes_fx_norm})['fx'].squeeze()
        stokes_ft_factor = norms['stokes_t'].inverse({'fx': stokes_ft_norm})['fx'].squeeze()
        # 速度项直接使用真实 vx，避免不一致的归一化文件导致量纲错误
        stokes_v_value = vx

        # 物理系数（cos 相位）
        drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
        sound_pressure = 20e-6 * (10 ** (sound_pressure_level / 20))
        A = sound_pressure / (1.225 * 340 * 2 * np.pi * 10000)
        stokes_amp = 2 * np.pi * 10000 * A

        # 合力（用户指定公式）
        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - stokes_amp * stokes_ft_factor * stokes_fx_factor)

        total_fx_cpu = arf_total + stokes_total
        fy = torch.zeros_like(total_fx_cpu)
        total_force = torch.stack([total_fx_cpu, fy], dim=1).to(positions_t.device)
        return total_force

def compute_force_using_physical_model(positions_t, velocities_t, t_scalar, phys_model):
    with torch.no_grad():
        x = positions_t[:, 0]
        vx = velocities_t[:, 0]
        t_vec = torch.full_like(x, float(t_scalar))
        total_fx = phys_model(x, vx, t_vec)
        fy = torch.zeros_like(total_fx)
        return torch.stack([total_fx, fy], dim=1)

def compute_force_components_using_trained_models(positions_t, velocities_t, t_scalar, models, norms):
    """返回分项 ARF 与 STOKES（均为CPU张量的一维: N）"""
    with torch.no_grad():
        x = positions_t[:, 0].detach().cpu()
        vx = velocities_t[:, 0].detach().cpu()
        t_vec = torch.full_like(x, float(t_scalar))

        x_norm_arf = norms['arf_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_arf = norms['arf_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        x_norm_stokes = norms['stokes_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_stokes = norms['stokes_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        vx_norm_stokes = norms['stokes_v'].transform({'vx': vx.unsqueeze(1)})['vx'].squeeze()

        arf_fx_norm = models['arf_x'](x_norm_arf.unsqueeze(1))
        arf_ft_norm = models['arf_t'](t_norm_arf.unsqueeze(1))
        stokes_fx_norm = models['stokes_x'](x_norm_stokes.unsqueeze(1))
        stokes_ft_norm = models['stokes_t'](t_norm_stokes.unsqueeze(1))
        stokes_fv_norm = models['stokes_v'](vx_norm_stokes.unsqueeze(1))

        arf_fx = norms['arf_x'].inverse({'fx': arf_fx_norm})['fx'].squeeze()
        arf_ft = norms['arf_t'].inverse({'fx': arf_ft_norm})['fx'].squeeze()
        stokes_fx_factor = norms['stokes_x'].inverse({'fx': stokes_fx_norm})['fx'].squeeze()
        stokes_ft_factor = norms['stokes_t'].inverse({'fx': stokes_ft_norm})['fx'].squeeze()
        stokes_v_value = norms['stokes_v'].inverse({'fx': stokes_fv_norm})['fx'].squeeze()

        drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
        sound_pressure = 20e-6 * (10 ** (sound_pressure_level / 20))
        A = sound_pressure / (1.225 * 340 * 2 * np.pi * 10000)
        stokes_amp = 2 * np.pi * 10000 * A

        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - stokes_amp * stokes_ft_factor * stokes_fx_factor)
        return arf_total, stokes_total

def compute_force_components_using_unified_model(positions_t, velocities_t, t_scalar, unified_model):
    with torch.no_grad():
        x = positions_t[:, 0].detach().cpu()
        vx = velocities_t[:, 0].detach().cpu()
        t_vec = torch.full_like(x, float(t_scalar))
        total_fx, arf_fx, stokes_fx = unified_model(x.unsqueeze(1), vx.unsqueeze(1), t_vec.unsqueeze(1))
        return arf_fx.squeeze().cpu(), stokes_fx.squeeze().cpu()

def compute_particle_forces_using_models(positions_t, velocities_t, mass_t, radii_t, dt, t_scalar, models, norms):
    """使用五个已训练模型计算受力并推进一步"""
    try:
        forces_t = compute_force_using_trained_models(positions_t, velocities_t, t_scalar, models, norms)
        # 计算加速度
        accelerations_t = forces_t / mass_t[:, None]
        # 更新速度和位置
        velocities_t = velocities_t + accelerations_t * dt
        positions_t = positions_t + velocities_t * dt
        return positions_t, velocities_t, (forces_t[:, 0], forces_t[:, 1])
    except Exception:
        return positions_t, velocities_t, None

def main():
    """主函数 - 直接计算，不运行动画"""
    global positions, velocities, mass, radii, initial_positions, domain_size
    
    print("=== 统一PINN模型粒子仿真（直接计算模式）===")
    
    # 设备与数值精度设置
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if use_cuda:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    # 初始化粒子
    domain_size = (0.034, 0.034)  # 34mm x 34mm
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=10000, domain_size=domain_size)
    initial_positions = positions_np.copy()

    # 转为 torch 张量并放到设备
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)

    # 预先分配 ones 向量用于生成 t 向量
    ones_vec = torch.ones(positions_t.shape[0], device=device)
    
    print(f"初始化了 {positions_t.shape[0]} 个粒子")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
    # 加载运行时物理统一模型（内部完成反归一化）
    print("\n正在加载物理统一模型…")
    phys_model = load_physical_unified_model(device)

    # 关闭 torch.compile，避免 Triton 依赖
    # 如果未来环境满足，可重新启用
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    
    print("物理统一模型加载完成")
    
    # 开始计算
    steps = 10000
    simulation_time = 0.0
    print(f"\n开始计算: {steps} 步，时间步长: {dt:.2e} s")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    
    # 计算开始时，先计算所有粒子当前的总力（t=0）用于尺度校验
    f0 = compute_force_using_physical_model(positions_t, velocities_t, 0.0, phys_model)[:, 0]
    print("初始时刻(t=0) 总力统计:")
    print(f"  Fx 范围: [{f0.min().item():.2e}, {f0.max().item():.2e}] N | 均值: {f0.mean().item():.2e} N")
    # 正式时间推进
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time

        # 记录更新前的位置和速度
        positions_before = positions_t.clone()
        velocities_before = velocities_t.clone()

        # 力学更新：使用物理统一模型（已反归一化）
        forces_t = compute_force_using_physical_model(positions_t, velocities_t, t, phys_model)
        accelerations_t = forces_t / mass_t[:, None]
        velocities_t = velocities_t + accelerations_t * dt
        positions_t = positions_t + velocities_t * dt

        # 可选调试与进度输出已移除，避免频繁打印
    
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"最终仿真时间: {simulation_time:.6f} s")
    
    # 重新设计的绘图方式：使用直方图显示粒子分布
    print("\n绘制粒子分布图...")
    
    # 将最终位置从设备转回 CPU numpy
    positions_final_np = positions_t.detach().cpu().numpy()
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：粒子位置对比
    ax1.scatter(initial_positions[:, 0] * 1000, initial_positions[:, 1] * 1000, 
               s=1, c='blue', alpha=0.5, label='Initial')
    ax1.scatter(positions_final_np[:, 0] * 1000, positions_final_np[:, 1] * 1000, 
               s=1, c='red', alpha=0.5, label='Final')
    ax1.set_xlim(0, domain_size[0] * 1000)
    ax1.set_ylim(0, domain_size[1] * 1000)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Particle Positions: Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：粒子密度分布（严格按照 comsol_visualizer.py 的方式）
    # 创建x轴网格（与 comsol_visualizer.py 一致）
    x_grid = np.linspace(0.0, 34.0, 100)  # 100个点用于平滑显示，覆盖0-34mm
    
    # 按照 comsol_visualizer.py 的方式计算浓度分布（高斯核密度估计）
    def calculate_concentration_distribution(positions, x_grid_mm):
        """按照 comsol_visualizer.py 的方式计算浓度分布 - 高斯核密度估计法"""
        if len(positions) == 0:
            return np.zeros_like(x_grid_mm)
        
        # 将粒子 x 坐标从米转换为毫米
        x_positions_mm = positions[:, 0] * 1000.0
        
        # 计算每个x_grid位置处的浓度
        concentrations = np.zeros_like(x_grid_mm)
        
        # 高斯核密度估计参数（与 comsol_visualizer.py 完全一致）
        sigma = 0.2  # 浓度分布的标准差（mm）
        
        for i, x_pos in enumerate(x_grid_mm):
            # 计算所有粒子到当前x位置的距离
            distances = np.abs(x_positions_mm - x_pos)
            
            # 浓度计算：距离越近，浓度越高
            # 使用高斯函数形式的浓度分布
            concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
            
            concentrations[i] = concentration
        
        return concentrations
    
    # 计算初始和最终浓度分布
    # 初始状态：所有粒子均匀分布，浓度分布为常数
    total_particles = len(initial_positions)
    initial_concentration = np.full_like(x_grid, total_particles / len(x_grid))  # 均匀分布
    
    # 最终状态：使用高斯核密度估计计算
    final_concentration = calculate_concentration_distribution(positions_final_np, x_grid)
    
    # 计算相对变化（与初始情况的对比）
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentration - initial_concentration) / initial_concentration * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 打印调试信息
    print(f"\n=== 浓度分布调试信息 ===")
    print(f"初始状态: 均匀分布，浓度 = {initial_concentration[0]:.3f} (常数)")
    print(f"最终浓度范围: {np.min(final_concentration):.3f} - {np.max(final_concentration):.3f}")
    print(f"初始浓度平均值: {np.mean(initial_concentration):.3f}")
    print(f"最终浓度平均值: {np.mean(final_concentration):.3f}")
    print(f"总粒子数: {total_particles}")
    print(f"相对变化范围: {np.min(relative_change):.3f}% - {np.max(relative_change):.3f}%")
    print(f"相对变化平均值: {np.mean(relative_change):.3f}%")
    
    # 始终显示相对变化（相对于均匀分布的百分比）
    print("显示相对变化（相对于均匀分布的百分比）")
    display_data = relative_change
    ylabel = 'Relative Change (%)'
    
    # 按照 comsol_visualizer.py 的方式绘制（高斯核密度估计 + 柱状图）
    # 先设置轴属性
    ax2.set_xlim(0.0, 34.0)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel(ylabel)
    ax2.set_title('Particle Density Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 创建柱状图（与 comsol_visualizer.py 一致）
    bars = ax2.bar(x_grid, display_data, width=0.34, align='center', alpha=0.8, 
                   color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
    
    # 添加平滑曲线（PCHIP 连接柱顶）
    from scipy.interpolate import PchipInterpolator
    try:
        pchip = PchipInterpolator(x_grid, display_data, extrapolate=False)
        x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
        y_smooth = pchip(x_smooth)
        curve_line, = ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line, = ax2.plot(x_grid, display_data, 'r-', linewidth=2, label='Smoothed')
    
    # 动态调整y轴范围
    if np.any(np.isfinite(display_data)):
        y_min = float(np.nanmin(display_data))
        y_max = float(np.nanmax(display_data))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 粒子分布统计 ===")
    print(f"初始粒子数: {len(initial_positions)}")
    print(f"最终粒子数: {len(positions_final_np)}")
    print(f"初始x范围: {initial_positions[:, 0].min()*1000:.2f} - {initial_positions[:, 0].max()*1000:.2f} mm")
    print(f"最终x范围: {positions_final_np[:, 0].min()*1000:.2f} - {positions_final_np[:, 0].max()*1000:.2f} mm")
    
    print("\n仿真完成！")

if __name__ == "__main__":
    main()
