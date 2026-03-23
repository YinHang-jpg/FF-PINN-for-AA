# 修复Windows中文编码问题
import sys
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import numpy as np
import torch
import torch.nn as nn

# 禁用matplotlib显示，避免在自动测试时弹出图片
import matplotlib
import matplotlib.pyplot as plt

import json
import os
import time
import contextlib
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

# 导入初始化函数
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level, compute_sound_field

# 导入各个PINN模型与归一化器（参考 PINN_preview.py）
from mechanisms.UNIFIED_PINN import (
    load_individual_models,
    UnifiedNormalizer,
    PhysicalUnifiedModel,
)

# 碰撞处理函数直接实现在本文件中，只考虑x方向碰撞

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理参数
USE_STOKES_DRAG = True  # 是否使用Stokes阻力
USE_ARF = True  # 是否使用声辐射力
dt = 1e-6  # 时间步长 (s)
gravity = 0.0  # 重力加速度 (m/s²)，已禁用y方向重力
particle_density = 2000  # 粒子密度 (kg/m³)
air_density = 1.225  # 空气密度 (kg/m³)
viscosity = 1.8e-5  # 空气动力粘度 (Pa·s)
fixed_diameter = 2e-6  # 固定粒子直径 (m)
fixed_cunningham = 1.083  # 固定Cunningham修正因子

# 碰撞处理参数
COLLISION_CHECK_INTERVAL = 1  # 每个时步都检查碰撞


def load_physical_unified_model(device):
    """加载物理统一模型
    
    通过环境变量 PINN_FREQ_FOLDER 可以指定使用哪个频率的模型
    例如: PINN_FREQ_FOLDER='freq_8k' 会从 PINN/freq_8k/ 加载模型
    """
    # 检查是否指定了特定的频率文件夹（通过环境变量）
    freq_folder = os.environ.get('PINN_FREQ_FOLDER', None)
    
    if freq_folder:
        model_base_path = os.path.join('PINN', freq_folder)
        print(f"  使用频率特定模型: {model_base_path}")
    else:
        model_base_path = 'PINN'
        print(f"  使用默认模型: {model_base_path}")
    
    models = load_individual_models(device, model_base_path)
    # 统一归一化器聚合各子模型的归一化范围/统计
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': os.path.join(model_base_path, 'arf_model_x_normalization_params.json'),
        'arf_t_norm': os.path.join(model_base_path, 'arf_model_t_normalization_params.json'),
        'stokes_x_norm': os.path.join(model_base_path, 'stokes_model_x_normalization_params.json'),
        'stokes_t_norm': os.path.join(model_base_path, 'stokes_model_t_normalization_params.json'),
        'stokes_v_norm': os.path.join(model_base_path, 'stokes_model_v_normalization_params.json'),
    }
    unified_norm.load_from_individual_models(model_paths, device='cpu')
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
    return phys_model


def compute_force_using_physical_model(positions_t, velocities_t, t_scalar, phys_model, use_cuda: bool = False):
    """仅计算并返回 Fx（一维向量）。
    为了性能只返回 x 方向力；y 方向重力在主循环中以常量加速度直接施加。
    """
    with torch.inference_mode():
        x = positions_t[:, 0]
        vx = velocities_t[:, 0]
        t_vec = torch.full_like(x, float(t_scalar))
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fx = phys_model(x, vx, t_vec)
        else:
            fx = phys_model(x, vx, t_vec)
        return fx.float() if fx.dtype != torch.float32 else fx


def handle_collisions(positions_t, velocities_t, radii_t, mass_t, particle_count_np, device):
    """
    简化碰撞处理：只考虑x方向的碰撞（完全非弹性碰撞）
    
    :param positions_t: 粒子位置张量 (N, 2)
    :param velocities_t: 粒子速度张量 (N, 2)
    :param radii_t: 粒子半径张量 (N,)
    :param mass_t: 粒子质量张量 (N,)
    :param particle_count_np: 每个粒子由多少个原始粒子组成 (N,)
    :param device: torch设备
    :return: 更新后的位置、速度、半径、质量张量，粒子计数数组，以及碰撞数量
    """
    # 转换为numpy数组
    positions_np = positions_t.detach().cpu().numpy()
    velocities_np = velocities_t.detach().cpu().numpy()
    radii_np = radii_t.detach().cpu().numpy()
    mass_np = mass_t.detach().cpu().numpy()
    
    # 记录初始粒子数
    initial_count = len(positions_np)
    
    if initial_count < 2:
        return positions_t, velocities_t, radii_t, mass_t, particle_count_np, 0
    
    # 只考虑x方向的碰撞：使用迭代方法，每次合并后重新检查
    # 记录初始粒子数用于计算碰撞次数
    collisions_count = 0
    
    # 迭代处理碰撞，直到没有新的碰撞发生
    max_iterations = 100  # 防止无限循环
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        current_count = len(positions_np)
        
        if current_count < 2:
            break
        
        # 按x坐标排序
        x_positions = positions_np[:, 0]
        sorted_indices = np.argsort(x_positions)
        
        collision_pairs = []
        
        # 检查相邻粒子对（只考虑x方向距离）
        # 对于一维x方向碰撞，如果粒子已按x坐标排序，只需检查相邻粒子对即可
        # 因为如果粒子A和粒子C重叠，而粒子B在中间，那么B一定和A或C重叠
        for idx in range(len(sorted_indices) - 1):
            i = sorted_indices[idx]
            j = sorted_indices[idx + 1]
            
            # 计算x方向距离（由于已排序，j的x坐标 >= i的x坐标）
            x_dist = x_positions[j] - x_positions[i]
            sum_radii = radii_np[i] + radii_np[j]
            
            # 如果x方向距离小于半径和，则发生碰撞
            # 注意：x_dist必须严格小于sum_radii才算碰撞（等于时刚好接触，不算碰撞）
            if x_dist < sum_radii:
                collision_pairs.append((i, j))
        
        # 如果没有碰撞，退出循环
        if not collision_pairs:
            break
        
        # 按质量从大到小排序，优先处理大粒子（减少后续冲突）
        collision_pairs.sort(key=lambda p: mass_np[p[0]] + mass_np[p[1]], reverse=True)
        
        # 标记要删除的粒子
        to_delete = set()
        
        # 处理碰撞对
        for i, j in collision_pairs:
            # 跳过已经被合并的粒子
            if i in to_delete or j in to_delete:
                continue
            
            # 计算合并后的属性（完全非弹性碰撞）
            total_mass = mass_np[i] + mass_np[j]
            mass_i = mass_np[i]
            mass_j = mass_np[j]
            
            # 只更新x坐标为质心x坐标，y坐标保持第一个粒子的y坐标不变
            center_of_mass_x = (positions_np[i, 0] * mass_i + positions_np[j, 0] * mass_j) / total_mass
            positions_np[i, 0] = center_of_mass_x
            # y坐标保持不变
            
            # 只更新x方向速度为动量守恒的速度，y方向速度保持第一个粒子的y速度
            merged_vx = (velocities_np[i, 0] * mass_i + velocities_np[j, 0] * mass_j) / total_mass
            velocities_np[i, 0] = merged_vx
            # y方向速度保持不变
            
            # 体积守恒的半径（假设三维球体）
            merged_radius = np.cbrt(radii_np[i]**3 + radii_np[j]**3)
            radii_np[i] = merged_radius
            
            # 更新质量和粒子计数
            mass_np[i] = total_mass
            particle_count_np[i] = particle_count_np[i] + particle_count_np[j]
            
            # 标记第二个粒子为删除
            to_delete.add(j)
        
        # 删除被标记的粒子
        if to_delete:
            keep_mask = np.ones(current_count, dtype=bool)
            keep_mask[list(to_delete)] = False
            positions_np = positions_np[keep_mask]
            velocities_np = velocities_np[keep_mask]
            radii_np = radii_np[keep_mask]
            mass_np = mass_np[keep_mask]
            particle_count_np = particle_count_np[keep_mask]
    
    # 计算碰撞数量（合并的粒子数）
    final_count = len(positions_np)
    collisions_count = initial_count - final_count
    
    # 转回torch张量
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
    
    return positions_t, velocities_t, radii_t, mass_t, particle_count_np, collisions_count


def main():
    """主函数 - 直接计算，考虑粒子碰撞"""
    global positions, velocities, mass, radii, initial_positions, domain_size
    
    print("=== 统一PINN模型粒子仿真（含碰撞处理）===")
    
    # 设备与数值精度设置
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if use_cuda:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    # 初始化粒子（使用 initialization/particle_initialization.py）
    domain_size = (0.034, 0.034)  # 34mm x 34mm
    # 使用 linear 模式：水平均匀分布，x坐标均匀分布，y坐标固定在域中心
    # 注意：频率会通过 sound_source_standing.py 中的 frequency 变量动态变化
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(
        N=14000, domain_size=domain_size, diameter=fixed_diameter, 
        density=particle_density, init_mode='linear'
    )
    initial_positions = positions_np.copy()
    initial_count = len(positions_np)

    # 转为 torch 张量并放到设备
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
    
    # 初始化粒子计数数组（每个粒子初始计数为1，表示由1个原始粒子组成）
    particle_count_np = np.ones(len(positions_np), dtype=np.int32)

    print(f"初始化了 {positions_t.shape[0]} 个粒子")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
    # 加载运行时物理统一模型（内部完成反归一化）
    print("\n正在加载物理统一模型…")
    phys_model = load_physical_unified_model(device)
    try:
        phys_model.eval()
    except Exception:
        pass
    
    print("物理统一模型加载完成")
    
    # 开始计算
    # 从环境变量或模型路径推断频率
    freq_folder = os.environ.get('PINN_FREQ_FOLDER', None)
    if freq_folder:
        import re
        freq_match = re.search(r'freq_(\d+)k', freq_folder)
        if freq_match:
            actual_frequency = int(freq_match.group(1)) * 1000  # 转换为Hz
            print(f"从环境变量推断频率: {actual_frequency} Hz")
        else:
            actual_frequency = frequency  # 使用默认频率
            print(f"无法从环境变量推断频率，使用默认值: {actual_frequency} Hz")
    else:
        actual_frequency = frequency  # 使用默认频率
        print(f"未设置PINN_FREQ_FOLDER，使用默认频率: {actual_frequency} Hz")
    
    # 现在支持多周期仿真，使用周期性归一化
    period = 1.0 / actual_frequency  # 一个周期的时间
    max_simulation_time = 10 * period  # 仿真10个周期
    steps = 10000
    simulation_time = 0.0
    print(f"\n开始计算: {steps} 步，时间步长: {dt:.2e} s")
    print(f"频率: {actual_frequency} Hz, 周期: {period:.6f} s")
    print(f"最大仿真时间: {max_simulation_time:.6f} s ({max_simulation_time/period:.1f} 个周期)")
    print(f"碰撞检查间隔: 每 {COLLISION_CHECK_INTERVAL} 步检查一次")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    
    # 计算开始时，先计算所有粒子当前的总力（t=0）用于尺度校验（仅Fx）
    f0 = compute_force_using_physical_model(positions_t, velocities_t, 0.0, phys_model, use_cuda)
    print("初始时刻(t=0) 总力统计:")
    print(f"  Fx 范围: [{f0.min().item():.2e}, {f0.max().item():.2e}] N | 均值: {f0.mean().item():.2e} N")
    
    # 设置打印间隔：每0.1个周期打印一次
    print_interval = 0.1 * period  # 每0.1个周期打印一次
    next_print_time = print_interval
    
    # 碰撞统计
    total_collisions = 0
    collision_check_count = 0
    
    # 记录每个时步的碰撞次数（用于绘图）
    collisions_per_step = []  # 存储每个时步的碰撞次数
    
    # 在推演阶段关闭梯度与版本跟踪，减少开销
    with torch.inference_mode():
        # 预计算量（加速）
        inv_mass_1d = 1.0 / mass_t

        # 创建进度条
        pbar = tqdm(
            range(steps),
            desc="计算进度",
            unit="步",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        # 正式时间推进
        for step in pbar:
            # 推进时间
            simulation_time += dt
            t = simulation_time

            # 仅计算 Fx（模型前向按需启用AMP）
            fx_last = compute_force_using_physical_model(positions_t, velocities_t, t, phys_model, use_cuda)

            # 计算加速度并更新速度（逐列就地运算，避免临时张量）
            # ax = fx/m, ay = -g（常量）
            velocities_t[:, 0].add_(fx_last.mul(inv_mass_1d), alpha=dt)
            if gravity > 0:
                velocities_t[:, 1].add_(-gravity * dt)

            # 更新位置（就地）
            positions_t.add_(velocities_t, alpha=dt)

            # 每个时步都检查碰撞（使用优化的高效算法）
            # 更新质量倒数（因为质量可能改变）
            inv_mass_1d = 1.0 / mass_t
            
            # 处理碰撞（传入粒子计数数组）
            positions_t, velocities_t, radii_t, mass_t, particle_count_np, collisions = handle_collisions(
                positions_t, velocities_t, radii_t, mass_t, particle_count_np, device
            )
            
            # 更新质量倒数（碰撞后）
            inv_mass_1d = 1.0 / mass_t
            
            total_collisions += collisions
            collision_check_count += 1
            
            # 每个时步都记录碰撞次数
            collisions_per_step.append(collisions)
            
            # 更新进度条描述信息（减少更新频率以提升性能）
            if step % 100 == 0 or collisions > 0:
                current_particles = len(positions_t)
                pbar.set_postfix({
                    '粒子数': current_particles,
                    '碰撞': total_collisions,
                    '时间': f'{simulation_time:.4f}s'
                })
                
                if collisions > 0:
                    pbar.write(f"  步 {step+1}: 检测到 {collisions} 次碰撞，剩余粒子数: {len(positions_t)}")

            # 打印统计信息（避免二次模型前向，复用 fx_last）
            if simulation_time >= next_print_time:
                fx_current = fx_last
                vx_current = velocities_t[:, 0]
                vy_current = velocities_t[:, 1]
                _ = torch.sqrt(vx_current * vx_current + vy_current * vy_current)
                next_print_time += print_interval
        
        # 关闭进度条
        pbar.close()
    
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"最终仿真时间: {simulation_time:.6f} s")
    print(f"\n=== 碰撞统计 ===")
    print(f"初始粒子数: {initial_count}")
    print(f"最终粒子数: {len(positions_t)}")
    print(f"总碰撞次数: {total_collisions}")
    print(f"碰撞检查次数: {collision_check_count}")
    print(f"粒子减少数量: {initial_count - len(positions_t)}")
    
    # 重新设计的绘图方式：使用直方图显示粒子分布
    print("\n绘制粒子分布图...")
    
    # 将最终位置从设备转回 CPU numpy
    positions_final_np = positions_t.detach().cpu().numpy()
    
    # 创建四张图：2x2布局
    fig = plt.figure(figsize=(16, 12))
    
    # 图1：初始粒子分布
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(initial_positions[:, 0] * 1000, initial_positions[:, 1] * 1000, 
               s=1, c='blue', alpha=0.5)
    ax1.set_xlim(0, domain_size[0] * 1000)
    ax1.set_ylim(0, domain_size[1] * 1000)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Initial Particle Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 图2：最终粒子分布（根据聚集次数设置大小）
    ax2 = plt.subplot(2, 2, 2)
    # 粒子大小与聚集次数成正比，使用对数缩放以便更好地可视化
    sizes = np.maximum(1, np.log1p(particle_count_np) * 10)  # log1p(x) = log(1+x)，避免log(0)
    scatter = ax2.scatter(positions_final_np[:, 0] * 1000, positions_final_np[:, 1] * 1000, 
               s=sizes, c=particle_count_np, cmap='Reds', alpha=0.6, 
               edgecolors='darkred', linewidths=0.3, vmin=1, vmax=max(particle_count_np.max(), 2))
    ax2.set_xlim(0, domain_size[0] * 1000)
    ax2.set_ylim(0, domain_size[1] * 1000)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel('Y Position (mm)')
    ax2.set_title('Final Particle Distribution (size ∝ cluster size)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Particle Count (cluster size)')
    
    # 右图：粒子密度分布（严格按照 comsol_visualizer.py 的方式）
    # 创建x轴网格（与 comsol_visualizer.py 完全一致）
    x_grid = np.linspace(0.0, 34.0, 100)  # 100个点用于平滑显示，覆盖0-34mm
    
    # 按照 comsol_visualizer.py 的方式计算浓度分布（高斯核密度估计）
    def calculate_concentration_distribution(positions, x_grid_mm, particle_counts=None):
        """
        按照 comsol_visualizer.py 的方式计算浓度分布 - 高斯核密度估计法
        如果提供了 particle_counts，则每个粒子按其计数加权（考虑聚集）
        """
        if len(positions) == 0:
            return np.zeros_like(x_grid_mm)
        
        # 将粒子 x 坐标从米转换为毫米
        x_positions_mm = positions[:, 0] * 1000.0
        
        # 如果没有提供计数，默认每个粒子计数为1
        if particle_counts is None:
            particle_counts = np.ones(len(positions))
        
        # 计算每个x_grid位置处的浓度
        concentrations = np.zeros_like(x_grid_mm)
        
        # 高斯核密度估计参数（与 comsol_visualizer.py 完全一致）
        sigma = 0.2  # 浓度分布的标准差（mm）
        
        for i, x_pos in enumerate(x_grid_mm):
            # 计算所有粒子到当前x位置的距离
            distances = np.abs(x_positions_mm - x_pos)
            
            # 浓度计算：距离越近，浓度越高
            # 使用高斯函数形式的浓度分布，并按粒子计数加权
            # 如果一个粒子团由20个粒子组成，它贡献20倍的浓度
            concentration = np.sum(particle_counts * np.exp(-distances**2 / (2 * sigma**2)))
            
            concentrations[i] = concentration
        
        return concentrations
    
    # 计算初始和最终浓度分布
    # 初始：每个粒子计数为1
    initial_particle_counts = np.ones(len(initial_positions), dtype=np.int32)
    initial_concentrations = calculate_concentration_distribution(initial_positions, x_grid, initial_particle_counts)
    
    # 最终：使用实际的粒子计数（考虑聚集）
    final_concentrations = calculate_concentration_distribution(positions_final_np, x_grid, particle_count_np)
    
    # 计算相对变化（与 comsol_visualizer.py 完全一致）
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 打印调试信息（与 comsol_visualizer.py 格式一致）
    print(f"\n[验证] 基于距离的浓度统计：t=0 vs 结束时")
    print(f"  实际粒子数: {len(initial_positions)} -> {len(positions_final_np)}")
    print(f"  等效粒子数（考虑聚集）: {len(initial_positions)} -> {particle_count_np.sum()}")
    print("  位置(mm)                初始浓度    最终浓度    相对变化(%)")
    
    # 选择关键位置进行显示（与 comsol_visualizer.py 一致）
    key_positions = [0.0, 8.5, 17.0, 25.5, 34.0]
    for pos in key_positions:
        # 找到最接近的x_grid索引
        idx = np.argmin(np.abs(x_grid - pos))
        c0 = initial_concentrations[idx]
        cT = final_concentrations[idx]
        change = relative_change[idx]
        print(f"  {pos:6.1f}                    {c0:8.3f}    {cT:8.3f}    {change:+8.1f}")
    
    # 计算整体统计
    avg_initial = np.mean(initial_concentrations)
    avg_final = np.mean(final_concentrations)
    avg_change = np.mean(relative_change)
    
    print("  ----------------------------------------------")
    print(f"  平均浓度                 {avg_initial:8.3f}    {avg_final:8.3f}    {avg_change:+8.1f}")
    
    # 图3：粒子密度分布（相对变化）
    ax3 = plt.subplot(2, 2, 3)
    # 先设置轴属性（与 comsol_visualizer.py 完全一致）
    ax3.set_xlim(0.0, 34.0)
    ax3.set_ylim(-5, 5)  # 与 comsol_visualizer.py 一致的y轴范围
    ax3.set_xlabel('x (mm)')
    ax3.set_ylabel('Relative Change (%)')
    ax3.set_title('Particle Density Distribution (Relative Change)')
    ax3.grid(True, alpha=0.3)
    
    # 创建柱状图（与 comsol_visualizer.py 完全一致）
    bars = ax3.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, 
                   color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
    
    # 添加平滑曲线（PCHIP 连接柱顶，与 comsol_visualizer.py 完全一致）
    try:
        pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
        x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
        y_smooth = pchip(x_smooth)
        curve_line, = ax3.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line, = ax3.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
    
    # 动态调整y轴范围（与 comsol_visualizer.py 完全一致）
    if np.any(np.isfinite(relative_change)):
        y_min = float(np.nanmin(relative_change))
        y_max = float(np.nanmax(relative_change))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax3.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax3.legend()
    
    # 图4：每个时步的碰撞趋势（直接绘制每个时步的碰撞次数）
    ax4 = plt.subplot(2, 2, 4)
    if len(collisions_per_step) > 0:
        steps_arr = np.arange(1, len(collisions_per_step) + 1, dtype=int)
        values = np.asarray(collisions_per_step, dtype=int)
        
        # 直接绘制每个时步的碰撞次数，用竖直线连接到x轴，并在顶点标出
        ax4.vlines(steps_arr, 0, values, colors='crimson', alpha=0.4, linewidth=1.0)
        ax4.scatter(steps_arr, values, s=12, color='crimson', alpha=0.8, zorder=3, label='Collisions per step')
        
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Collisions per Step')
        ax4.set_title('Collision Trend')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No collision data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Collision Trend')
    
    plt.tight_layout()
    
    # 保存图片
    output_filename = 'clustering_results.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {output_filename}")
    
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 粒子分布统计 ===")
    print(f"初始粒子数: {len(initial_positions)}")
    print(f"最终粒子数: {len(positions_final_np)}")
    print(f"初始x范围: {initial_positions[:, 0].min()*1000:.2f} - {initial_positions[:, 0].max()*1000:.2f} mm")
    print(f"最终x范围: {positions_final_np[:, 0].min()*1000:.2f} - {positions_final_np[:, 0].max()*1000:.2f} mm")
    
    print("\n仿真完成！")

    # 自动保存x_grid与relative_change到 'density_curve_clustering.txt'（列堆叠存储、浮点数）
    np.savetxt('density_curve_clustering.txt', np.vstack([x_grid, relative_change]).T)
    
    # 保存每个时步的碰撞次数到 'collisions_per_step.txt'
    if len(collisions_per_step) > 0:
        steps_arr = np.arange(1, len(collisions_per_step) + 1, dtype=int)
        collisions_arr = np.asarray(collisions_per_step, dtype=int)
        # 保存为两列：时步，碰撞次数
        np.savetxt('collisions_per_step.txt', np.vstack([steps_arr, collisions_arr]).T, 
                   fmt='%d', header='step collisions', comments='')
        print(f"碰撞数据已保存至: collisions_per_step.txt (共 {len(collisions_per_step)} 个时步)")


if __name__ == "__main__":
    main()

