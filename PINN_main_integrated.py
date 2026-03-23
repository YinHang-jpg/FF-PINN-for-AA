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
# dt 将根据频率动态计算，确保所有频率经历相同的周期数
gravity = 9.81  # 重力加速度 (m/s²)
particle_density = 2000  # 粒子密度 (kg/m³)
air_density = 1.225  # 空气密度 (kg/m³)
viscosity = 1.8e-5  # 空气动力粘度 (Pa·s)
fixed_diameter = 2e-6  # 固定粒子直径 (m)
fixed_cunningham = 1.083  # 固定Cunningham修正因子
 
 
def load_physical_unified_model(device, model_folder=None):
    """加载物理统一模型
   
    Args:
        device: torch设备
        model_folder: 模型文件夹名称（例如 'freq_8k' 或 'density/density_2000'），如果为None则使用默认PINN文件夹
    """
    # 确定模型路径前缀
    if model_folder:
        model_base_path = os.path.join('PINN', model_folder)
    else:
        model_base_path = 'PINN'
   
    print(f"\n{'='*70}")
    print(f"正在加载物理统一模型...")
    print(f"模型路径: {model_base_path}")
    print(f"{'='*70}\n")
   
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
 
        # 物理系数（修正斯托克斯阻力公式）
        mu = 1.86e-5
        cunningham = 1.0817
        diameter = 2e-6
        drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
        omega = 2.0 * np.pi * frequency
       
        # 动态计算振幅 A（根据频率和声压级）
        reference_pressure = 20e-6  # Pa
        sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
        rho_0 = 1.225  # 空气密度 (kg/m³)
        c_0 = 340      # 声速 (m/s)
        A = sound_pressure / (rho_0 * c_0 * omega)
       
        # 计算粒子质量（球形粒子，密度2000 kg/m³）
        particle_density = 2000.0
        radius = diameter / 2.0
        particle_mass = (4.0/3.0) * np.pi * (radius**3) * particle_density
       
        # 计算粒子响应时间：tau_p = mass / drag_coeff
        tau_p = particle_mass / drag_coeff
       
        # 计算频率响应衰减因子（考虑粒子惯性）
        # 高频时粒子无法跟随快速振荡的气流
        response_factor = 1.0 / (1.0 + (omega * tau_p)**2)
       
        # 气流速度：u_air = -omega*A*cos(kx)*cos(omega*t)
        u_air = -omega * A * stokes_fx_factor * stokes_ft_factor
       
        # 应用频率响应衰减到振荡气流速度
        u_air_eff = u_air * response_factor
       
        # 合力（修正公式）
        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - u_air_eff)
 
        total_fx_cpu = arf_total + stokes_total
        fy = torch.zeros_like(total_fx_cpu)
        total_force = torch.stack([total_fx_cpu, fy], dim=1).to(positions_t.device)
        return total_force
 
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
 
        # 物理系数（与其他函数保持一致）
        mu = 1.86e-5
        cunningham = 1.0817
        diameter = 2e-6
        drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
        omega = 2.0 * np.pi * frequency
       
        # 动态计算振幅 A（根据频率和声压级）
        reference_pressure = 20e-6  # Pa
        sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
        rho_0 = 1.225  # 空气密度 (kg/m³)
        c_0 = 340      # 声速 (m/s)
        A = sound_pressure / (rho_0 * c_0 * omega)
       
        # 计算粒子质量（球形粒子，密度2000 kg/m³）
        particle_density = 2000.0
        radius = diameter / 2.0
        particle_mass = (4.0/3.0) * np.pi * (radius**3) * particle_density
       
        # 计算粒子响应时间：tau_p = mass / drag_coeff
        tau_p = particle_mass / drag_coeff
       
        # 计算频率响应衰减因子（考虑粒子惯性）
        # 高频时粒子无法跟随快速振荡的气流
        response_factor = 1.0 / (1.0 + (omega * tau_p)**2)
       
        # 气流速度：u_air = -omega*A*cos(kx)*cos(omega*t)
        u_air = -omega * A * stokes_fx_factor * stokes_ft_factor
       
        # 应用频率响应衰减到振荡气流速度
        u_air_eff = u_air * response_factor
 
        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - u_air_eff)
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
       
        # 添加重力（如果需要）
        if gravity > 0:
            gravity_force = torch.zeros_like(forces_t)
            gravity_force[:, 1] = -mass_t * gravity  # 重力向下（y方向负）
            forces_t = forces_t + gravity_force
       
        # 计算加速度
        accelerations_t = forces_t / mass_t[:, None]
       
        # 更新速度和位置（使用前向欧拉法）
        velocities_t = velocities_t + accelerations_t * dt
        positions_t = positions_t + velocities_t * dt
       
        return positions_t, velocities_t, (forces_t[:, 0], forces_t[:, 1])
    except Exception:
        return positions_t, velocities_t, None
 
def main(total_steps=None, suppress_output=False):
    """主函数 - 直接计算，不运行动画
   
    Args:
        total_steps: 总仿真步数（如果为None则使用默认的100周期×100步）
        suppress_output: 是否抑制输出（用于批量运行）
   
    Returns:
        tuple: (x_grid, relative_change) 用于后续分析
    """
    global positions, velocities, mass, radii, initial_positions, domain_size
   
    if not suppress_output:
        print("=== 统一PINN模型粒子仿真（直接计算模式）===")
        print(f"当前配置频率: {frequency} Hz ({frequency/1000:.0f} kHz)")
        print(f"当前粒子密度: {particle_density} kg/m³")
   
    # 确定要使用的模型文件夹
    # 优先级：1. 密度文件夹（PINN_DENSITY_FOLDER） 2. 频率文件夹（PINN_FREQ_FOLDER） 3. 自动推断 4. 默认PINN文件夹
   
    model_folder = None
   
    # 首先检查是否指定了密度文件夹
    density_folder = os.environ.get('PINN_DENSITY_FOLDER', None)
    if density_folder:
        model_folder = os.path.join('density', density_folder)
        if not suppress_output:
            print(f"使用环境变量指定的密度文件夹: PINN/{model_folder}")
    else:
        # 如果没有密度文件夹，检查频率文件夹
        freq_folder = os.environ.get('PINN_FREQ_FOLDER', None)
       
        if not freq_folder:
            # 自动根据当前频率推断文件夹
            freq_label = f"{frequency//1000}k"
            auto_freq_folder = f"freq_{freq_label}"
            freq_folder_path = os.path.join('PINN', auto_freq_folder)
           
            if os.path.exists(freq_folder_path):
                freq_folder = auto_freq_folder
                if not suppress_output:
                    print(f"自动检测到频率文件夹: {freq_folder}")
            else:
                if not suppress_output:
                    print(f"警告：未找到频率文件夹 {freq_folder_path}")
                    print(f"将使用默认 PINN/ 文件夹")
                freq_folder = None
        else:
            if not suppress_output:
                print(f"使用环境变量指定的频率文件夹: {freq_folder}")
       
        model_folder = freq_folder
   
    # 验证模型文件是否存在
    if model_folder:
        model_base_path = os.path.join('PINN', model_folder)
        required_files = [
            'arf_model_x.pth',
            'arf_model_t.pth',
            'stokes_model_x.pth',
            'stokes_model_t.pth',
            'stokes_model_v.pth'
        ]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_base_path, file)):
                missing_files.append(file)
       
        if missing_files:
            if not suppress_output:
                print(f"\n错误：以下模型文件缺失于 {model_base_path}:")
                for file in missing_files:
                    print(f"  - {file}")
                print("\n请确保已完成该频率/密度下的模型训练。")
            return None, None
        else:
            if not suppress_output:
                print(f"✓ 所有必需的模型文件已找到于: {model_base_path}")
   
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
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=100000, domain_size=domain_size)
    initial_positions = positions_np.copy()
 
    # 转为 torch 张量并放到设备
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
 
    # 预先分配 ones 向量用于生成 t 向量
    ones_vec = torch.ones(positions_t.shape[0], device=device)
   
    if not suppress_output:
        print(f"初始化了 {positions_t.shape[0]} 个粒子")
        print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
   
    # 加载运行时物理统一模型（内部完成反归一化）
    if not suppress_output:
        print("\n正在加载物理统一模型…")
    phys_model = load_physical_unified_model(device, model_folder)
    try:
        phys_model.eval()
    except Exception:
        pass
 
    # 关闭 torch.compile，避免 Triton 依赖
    # 如果未来环境满足，可重新启用
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
   
    if not suppress_output:
        print("物理统一模型加载完成")
   
    # 开始计算
    # 现在支持多周期仿真，使用周期性归一化
    period = 1.0 / frequency  # 一个周期的时间
   
    if total_steps is None:
        # 使用默认配置
        num_cycles = 100  # 固定仿真100个周期
        steps_per_cycle = 100  # 每个周期100步
        steps = num_cycles * steps_per_cycle
    else:
        # 使用指定的总步数
        steps = total_steps
        # 假设每周期100步来计算周期数
        steps_per_cycle = 100
        num_cycles = steps / steps_per_cycle
   
    # 根据周期数计算总步数和时间步长
    max_simulation_time = num_cycles * period
    dt = max_simulation_time / steps  # 动态计算时间步长
   
    simulation_time = 0.0
    if not suppress_output:
        print(f"\n开始计算: {steps} 步，时间步长: {dt:.2e} s")
        print(f"频率: {frequency} Hz, 周期: {period:.6f} s")
        print(f"仿真周期数: {num_cycles} 个周期")
        print(f"每周期步数: {steps_per_cycle} 步")
        print(f"总仿真时间: {max_simulation_time:.6f} s")
   
    # 整体计算时间计时
    total_start_time = time.perf_counter()
   
    # 计算开始时，先计算所有粒子当前的总力（t=0）用于尺度校验（仅Fx）
    f0 = compute_force_using_physical_model(positions_t, velocities_t, 0.0, phys_model, use_cuda)
    if not suppress_output:
        print("初始时刻(t=0) 总力统计:")
        print(f"  Fx 范围: [{f0.min().item():.2e}, {f0.max().item():.2e}] N | 均值: {f0.mean().item():.2e} N")
   
    # 设置打印间隔：每0.1个周期打印一次
    print_interval = 0.1 * period  # 每0.1个周期打印一次
    next_print_time = print_interval
   
    # 在推演阶段关闭梯度与版本跟踪，减少开销
    with torch.inference_mode():
        # 预计算量（加速）
        inv_mass_1d = 1.0 / mass_t
 
        # 正式时间推进（标准前向欧拉法）
        for step in range(steps):
            # 标准前向欧拉法：
            # 1. 用当前时刻t的力 F(x(t), v(t), t)
            # 2. 更新位置：x(t+dt) = x(t) + v(t) * dt  （用旧速度）
            # 3. 更新速度：v(t+dt) = v(t) + F(t)/m * dt （用旧力）
            # 4. 时间推进：t = t + dt
           
            # 当前时刻
            t = simulation_time
           
            # 第1步：计算当前时刻的力 F(x(t), v(t), t)
            fx_current = compute_force_using_physical_model(positions_t, velocities_t, t, phys_model, use_cuda)
           
            # 第2步：用当前速度（旧速度）更新位置
            # x(t+dt) = x(t) + v(t) * dt
            positions_t.add_(velocities_t, alpha=dt)
           
            # 第3步：用当前力更新速度
            # v(t+dt) = v(t) + a(t) * dt，其中 a(t) = F(t)/m
            velocities_t[:, 0].add_(fx_current.mul(inv_mass_1d), alpha=dt)
            if gravity > 0:
                velocities_t[:, 1].add_(-gravity * dt)
           
            # 第4步：时间推进
            simulation_time += dt
 
            # 打印统计信息（避免二次模型前向，复用已计算的力）
            if simulation_time >= next_print_time:
                # fx_current 已在上面计算，无需重复
                vx_current = velocities_t[:, 0]
                vy_current = velocities_t[:, 1]
                _ = torch.sqrt(vx_current * vx_current + vy_current * vy_current)
                next_print_time += print_interval
   
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
   
    if not suppress_output:
        print(f"\n=== 计算完成 ===")
        print(f"总步数: {steps}")
        print(f"时间步长: {dt:.2e} s")
        print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
        print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
        print(f"最终仿真时间: {simulation_time:.6f} s")
   
    # 重新设计的绘图方式：使用直方图显示粒子分布
    if not suppress_output:
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
    # 创建x轴网格（与 comsol_visualizer.py 完全一致）
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
   
    # 计算初始和最终浓度分布（与 comsol_visualizer.py 完全一致）
    initial_concentrations = calculate_concentration_distribution(initial_positions, x_grid)
    final_concentrations = calculate_concentration_distribution(positions_final_np, x_grid)
   
    # 计算相对变化（与 comsol_visualizer.py 完全一致）
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
   
    # 打印调试信息（与 comsol_visualizer.py 格式一致）
    if not suppress_output:
        print(f"\n[验证] 基于距离的浓度统计：t=0 vs 结束时")
        print(f"  粒子总数: {len(initial_positions)}")
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
   
    # 按照 comsol_visualizer.py 的方式绘制（高斯核密度估计 + 柱状图）
    # 先设置轴属性（与 comsol_visualizer.py 完全一致）
    ax2.set_xlim(0.0, 34.0)
    ax2.set_ylim(-5, 5)  # 与 comsol_visualizer.py 一致的y轴范围
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('Relative Change (%)')
    ax2.set_title('Particle Density Distribution')
    ax2.grid(True, alpha=0.3)
   
    # 创建柱状图（与 comsol_visualizer.py 完全一致）
    bars = ax2.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8,
                   color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
   
    # 添加平滑曲线（PCHIP 连接柱顶，与 comsol_visualizer.py 完全一致）
    try:
        pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
        x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
        y_smooth = pchip(x_smooth)
        curve_line, = ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line, = ax2.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
   
    # 动态调整y轴范围（与 comsol_visualizer.py 完全一致）
    if np.any(np.isfinite(relative_change)):
        y_min = float(np.nanmin(relative_change))
        y_max = float(np.nanmax(relative_change))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
   
    ax2.legend()
   
    plt.tight_layout()
   
    # 图像输出处理：支持环境变量指定保存路径，或显示/关闭
    save_image_path = os.environ.get('PINN_SAVE_IMAGE_PATH', None)
    if save_image_path:
        # 通过环境变量指定了保存路径
        plt.savefig(save_image_path, dpi=150, bbox_inches='tight')
        if not suppress_output:
            print(f'图像已保存到: {save_image_path}')
        plt.close(fig)
    elif not suppress_output:
        # 非抑制模式：显示图像
        plt.show()
    else:
        # 抑制模式：关闭图像
        plt.close(fig)
   
    # 打印统计信息
    if not suppress_output:
        print(f"\n=== 粒子分布统计 ===")
        print(f"初始粒子数: {len(initial_positions)}")
        print(f"最终粒子数: {len(positions_final_np)}")
        print(f"初始x范围: {initial_positions[:, 0].min()*1000:.2f} - {initial_positions[:, 0].max()*1000:.2f} mm")
        print(f"最终x范围: {positions_final_np[:, 0].min()*1000:.2f} - {positions_final_np[:, 0].max()*1000:.2f} mm")
       
        print("\n仿真完成！")
 
    # 自动保存x_grid与relative_change到 'density_curve_PINN.txt'（列堆叠存储、浮点数）
    # 如果指定了保存路径或非抑制模式，则保存数据文件
    if save_image_path or not suppress_output:
        np.savetxt('density_curve_PINN.txt', np.vstack([x_grid, relative_change]).T)
   
    # 返回结果供外部脚本使用
    return x_grid, relative_change
 
if __name__ == "__main__":
    main()