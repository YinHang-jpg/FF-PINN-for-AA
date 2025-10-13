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

# 导入统一PINN模型
from mechanisms.UNIFIED_PINN import UnifiedPINN, UnifiedNormalizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理参数
USE_STOKES_DRAG = True  # 是否使用Stokes阻力
USE_ARF = True  # 是否使用声辐射力
dt = 1e-6  # 时间步长 (s)
gravity = 9.81  # 重力加速度 (m/s²)
particle_density = 1000  # 粒子密度 (kg/m³)
air_density = 1.225  # 空气密度 (kg/m³)
viscosity = 1.8e-5  # 空气动力粘度 (Pa·s)
fixed_diameter = 2e-6  # 固定粒子直径 (m)
fixed_cunningham = 1.083  # 固定Cunningham修正因子

def calculate_concentration_distribution(positions, x_grid_mm):
    """
    计算浓度分布（与 validation/comsol_visualizer.py 一致的方式）：
    - 在毫米坐标系下计算
    - 使用高斯核（sigma=0.2 mm）对每个 x 网格点统计邻近粒子浓度

    Args:
        positions: 粒子位置（单位：米），形状 (N, 2)
        x_grid_mm: x 轴网格（单位：毫米）
    """
    if len(positions) == 0:
        return np.zeros_like(x_grid_mm)

    # 将粒子 x 坐标从米转换为毫米
    x_positions_mm = positions[:, 0] * 1000.0

    # 计算每个网格点的浓度（高斯核平滑）
    concentration = np.zeros_like(x_grid_mm)

    sigma_mm = 0.2  # 浓度分布的标准差（mm）
    two_sigma2 = 2.0 * (sigma_mm ** 2)

    for i, x_center_mm in enumerate(x_grid_mm):
        distances = np.abs(x_positions_mm - x_center_mm)
        weights = np.exp(-(distances ** 2) / two_sigma2)
        concentration[i] = np.sum(weights)

    return concentration

def compute_unified_force_using_pinn(positions_t, velocities_t, t_scalar, unified_model, unified_normalizer, ones_vec=None):
    """
    使用训练好的统一PINN模型计算粒子合力
    直接调用统一模型的forward方法，不调用合并前的五个模型
    """
    if unified_model is None or unified_normalizer is None:
        print("统一PINN模型未完全初始化，无法计算力")
        return None
    
    try:
        with torch.no_grad():
            x = positions_t[:, 0]
            vx = velocities_t[:, 0]
            if ones_vec is None:
                t_vec = torch.full_like(x, float(t_scalar))
            else:
                t_vec = ones_vec * float(t_scalar)

            # 归一化输入
            x_norm, vx_norm, t_norm = unified_normalizer.normalize_inputs(x, vx, t_vec)

            # 直接使用统一模型的forward方法（内部使用神经网络，不调用合并前的模型）
            total_fx, _arf_fx, _stokes_fx = unified_model(x_norm.unsqueeze(1), vx_norm.unsqueeze(1), t_norm.unsqueeze(1))

            # 总力反归一化（与ARF相同的归一化参数）
            total_fx_mu = unified_normalizer.stats['arf_fx_mu']
            total_fx_sigma = unified_normalizer.stats['arf_fx_sigma']
            total_fx = total_fx.squeeze() * total_fx_sigma + total_fx_mu

            fy_total = torch.zeros_like(total_fx)  # y方向为零

            return torch.stack([total_fx, fy_total], dim=1)
            
    except Exception as e:
        print(f"统一PINN模型计算力时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_particle_forces_using_unified_pinn(positions_t, velocities_t, mass_t, radii_t, dt, t_scalar, unified_model, unified_normalizer, ones_vec=None):
    """
    使用训练好的统一PINN模型计算粒子受力
    """
    if unified_model is None or unified_normalizer is None:
        print("统一PINN模型未完全初始化，无法计算力")
        return positions_t, velocities_t, None
    
    try:
        # 使用统一模型计算合力（torch 张量）
        forces_t = compute_unified_force_using_pinn(positions_t, velocities_t, t_scalar, unified_model, unified_normalizer, ones_vec=ones_vec)
        if forces_t is None:
            return positions_t, velocities_t, None

        # 计算加速度
        accelerations_t = forces_t / mass_t[:, None]

        # 更新速度和位置
        velocities_t = velocities_t + accelerations_t * dt
        positions_t = positions_t + velocities_t * dt

        return positions_t, velocities_t, (forces_t[:, 0], forces_t[:, 1])
        
    except Exception as e:
        print(f"统一PINN模型计算力时出错: {e}")
        print("将使用零力（无外力效应）")
        # 如果PINN计算失败，不施加任何力，保持原有运动
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
    
    # 加载统一PINN模型
    print("\n正在加载统一PINN模型...")
    
    # 初始化统一模型
    unified_model = UnifiedPINN(fourier_features=32)
    unified_model_path = 'PINN/unified_model.pth'
    if os.path.exists(unified_model_path):
        unified_model.load_state_dict(torch.load(unified_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载统一模型权重: {unified_model_path}")
    else:
        print(f"错误: 统一模型文件 {unified_model_path} 不存在")
        print("请先训练统一模型")
        return
    
    # 加载统一归一化参数
    unified_norm_path = 'PINN/unified_model_normalization_params.json'
    unified_normalizer = UnifiedNormalizer()
    if os.path.exists(unified_norm_path):
        with open(unified_norm_path, 'r') as f:
            params = json.load(f)
        unified_normalizer.stats = {
            'x_min': params.get('x_min', 0.0),
            'x_max': params.get('x_max', 0.1),
            'vx_min': params.get('vx_min', -0.1),
            'vx_max': params.get('vx_max', 0.1),
            't_min': params.get('t_min', 0.0),
            't_max': params.get('t_max', 1.0/frequency),
            'arf_fx_mu': params.get('arf_fx_mu', 0.0),
            'arf_fx_sigma': params.get('arf_fx_sigma', 1.0),
            'arf_ft_mu': params.get('arf_ft_mu', 0.0),
            'arf_ft_sigma': params.get('arf_ft_sigma', 1.0),
            'stokes_fx_mu': params.get('stokes_fx_mu', 0.0),
            'stokes_fx_sigma': params.get('stokes_fx_sigma', 1.0),
            'stokes_ft_mu': params.get('stokes_ft_mu', 0.0),
            'stokes_ft_sigma': params.get('stokes_ft_sigma', 1.0),
            'stokes_fv_mu': params.get('stokes_fv_mu', 0.0),
            'stokes_fv_sigma': params.get('stokes_fv_sigma', 1.0)
        }
        print(f"成功加载统一模型归一化参数: {unified_norm_path}")
    else:
        print(f"错误: 统一模型归一化参数文件 {unified_norm_path} 不存在")
        print("请先训练统一模型")
        return
    
    # 设置为评估模式
    # 移动到设备
    unified_model.eval()
    unified_model.to(device)

    # 关闭 torch.compile，避免 Triton 依赖
    # 如果未来环境满足，可重新启用
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    
    print("统一PINN模型初始化完成")
    
    # 开始计算
    steps = 10000
    simulation_time = 0.0
    print(f"\n开始计算: {steps} 步，时间步长: {dt:.2e} s")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time

        # 力学更新：使用统一PINN模型（保持在设备上）
        if use_cuda:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                positions_t, velocities_t, _ = compute_particle_forces_using_unified_pinn(
                    positions_t, velocities_t, mass_t, radii_t, dt, t, unified_model, unified_normalizer, ones_vec=ones_vec
                )
        else:
            positions_t, velocities_t, _ = compute_particle_forces_using_unified_pinn(
                positions_t, velocities_t, mass_t, radii_t, dt, t, unified_model, unified_normalizer, ones_vec=ones_vec
            )
        
        # 每1000步打印一次进度
        if (step + 1) % 1000 == 0:
            current_time = time.perf_counter()
            elapsed_so_far = current_time - total_start_time
            progress = (step + 1) / steps * 100
            print(f"进度: {step + 1}/{steps} ({progress:.1f}%) - 已用时: {elapsed_so_far:.2f}s")
    
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"最终仿真时间: {simulation_time:.6f} s")
    
    # 计算并显示粒子密度分布曲线
    print("\n计算粒子密度分布...")
    
    # 创建x轴网格（毫米，与 validation/comsol_visualizer.py 一致）
    x_grid = np.linspace(0.0, 34.0, 100)
    
    # 计算初始和最终浓度分布（毫米坐标系下）
    # 将最终位置从设备转回 CPU numpy
    positions_final_np = positions_t.detach().cpu().numpy()
    initial_concentration = calculate_concentration_distribution(initial_positions, x_grid)
    final_concentration = calculate_concentration_distribution(positions_final_np, x_grid)
    
    # 计算相对变化
    with np.errstate(divide='ignore', invalid='ignore'):
        # 与 validation/comsol_visualizer.py 相同：以初始浓度为分母，不加额外偏置
        relative_change = (final_concentration - initial_concentration) / initial_concentration * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 创建图形
    fig, (ax_particles, ax_density) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：粒子位置对比
    ax_particles.scatter(initial_positions[:, 0] * 1000, initial_positions[:, 1] * 1000, 
                        s=20, c='blue', alpha=0.6, label='Initial')
    ax_particles.scatter(positions_final_np[:, 0] * 1000, positions_final_np[:, 1] * 1000, 
                        s=20, c='red', alpha=0.6, label='Final')
    ax_particles.set_xlim(0, domain_size[0] * 1000)
    ax_particles.set_ylim(0, domain_size[1] * 1000)
    ax_particles.set_xlabel('X Position (mm)')
    ax_particles.set_ylabel('Y Position (mm)')
    ax_particles.set_title('Particle Positions: Initial vs Final')
    ax_particles.grid(True, alpha=0.3)
    ax_particles.legend()
    
    # 右图：密度分布曲线（参考comsol_visualizer.py）
    ax_density.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, 
                   color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
    
    # 添加平滑曲线
    try:
        pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
        x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
        y_smooth = pchip(x_smooth)
        ax_density.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        ax_density.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
    
    # 动态调整y轴范围
    if np.any(np.isfinite(relative_change)):
        y_min = float(np.nanmin(relative_change))
        y_max = float(np.nanmax(relative_change))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax_density.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax_density.set_xlim(0.0, 34.0)
    ax_density.set_xlabel('X Position (mm)')
    ax_density.set_ylabel('Relative Concentration Change (%)')
    ax_density.set_title('Particle Density Distribution')
    ax_density.grid(True, alpha=0.3)
    ax_density.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 密度分布统计 ===")
    print(f"初始平均浓度: {np.mean(initial_concentration):.3f}")
    print(f"最终平均浓度: {np.mean(final_concentration):.3f}")
    print(f"平均相对变化: {np.mean(relative_change):.1f}%")
    print(f"最大相对变化: {np.max(relative_change):.1f}%")
    print(f"最小相对变化: {np.min(relative_change):.1f}%")
    
    print("\n仿真完成！")

if __name__ == "__main__":
    main()
