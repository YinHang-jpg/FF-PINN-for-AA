import matplotlib.pyplot as plt

import matplotlib

import matplotlib.animation as animation
from initialization.particle_initialization import initialize_particles
import numpy as np
import psutil
import time
import threading
import os
import torch
import json
import sys

# 修复Windows中文编码问题
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())


# 修复Windows中文编码问题
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator

# 配置开关 - 使用ARF物理公式进行力预测，同时开启斯托克斯阻力
RUN_HEADLESS_BENCHMARK = True # 关闭渲染与动画，仅计算并在1000步后退出
USE_TRAVELING_WAVE = False
USE_TEST_PARTICLE = False
USE_ARF_FORCES = False  # 使用ARF物理公式计算所有粒子受力
USE_GRAVITY = False
USE_STOKES_DRAG = True # 开启斯托克斯阻力
USE_AGGLOMERATION = False
USE_BROWNIAN = False
USE_ACOUSTIC_WAKE = False  # 关闭，由ARF物理公式处理
USE_COLLISION = False  # 关闭碰撞处理

# 调试打印配置
TARGET_PARTICLE_INDEX = 0  # 要监控的粒子索引


# 仿真时间累计变量
simulation_time = 0.0  # 累计仿真时间
density_plot_saved = False  # 标记是否已经保存过密度图
last_save_time = 0.0  # 记录上次保存图片的时间

# 性能计时变量
step_timing_printed = False  # 标记是否已打印过第5帧的计时信息



# 导入声场计算（仅用于可视化）
if USE_TRAVELING_WAVE:
    from initialization.sound_source_traveling import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = False
else:
    from initialization.sound_source_standing import compute_sound_field, frequency, sound_pressure_level, spl_to_pressure
    IS_STANDING_WAVE = True
    # 动态计算振幅
    def get_amplitude():
        return 2 * spl_to_pressure(sound_pressure_level)

# 导入ARF物理公式计算模块
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf

# 导入斯托克斯阻力模块
from mechanisms.Stokes_drag import apply_stokes_drag, cunningham_correction_factor

# 仅用于打印：根据当前速度计算每个粒子的斯托克斯阻力（与 mechanisms/Stokes_drag.py 一致）
def compute_stokes_drag_force(positions, velocities, radii, viscosity=1.79e-5, fluid_velocity=None, lambda_g=6.5e-8, time=0.0, use_sound_velocity=True):
    N = len(positions)
    if fluid_velocity is None:
        if use_sound_velocity:
            # 使用声波影响下的空气速度
            from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
            fluid_velocity = compute_air_velocity_due_to_sound(positions, time)
        else:
            fluid_velocity = np.zeros((N, 2))
    drag_force = np.zeros_like(positions)
    for i in range(N):
        d_p = 2.0 * radii[i]
        C_c = cunningham_correction_factor(d_p, lambda_g)
        relative_velocity = velocities[i] - fluid_velocity[i]
        drag_coeff = 3.0 * np.pi * viscosity * d_p / C_c
        drag_force[i] = -drag_coeff * relative_velocity
    return drag_force


# 性能监控变量
performance_data = {
    'cpu_percent': 0.0,
    'memory_percent': 0.0,
    'memory_used_gb': 0.0,
    'fps': 0.0,
    'frame_time': 0.0,
    'particle_count': 0
}

# 性能监控函数
def update_performance_data():
    """更新性能数据"""
    while True:
        try:
            # CPU使用率
            performance_data['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率和具体使用量
            memory = psutil.virtual_memory()
            performance_data['memory_percent'] = memory.percent
            performance_data['memory_used_gb'] = memory.used / (1024**3)  # 转换为GB
            
            # 粒子数量
            performance_data['particle_count'] = len(positions) if 'positions' in globals() else 0
            
            time.sleep(0.5)  # 每0.5秒更新一次
        except:
            pass

# 启动性能监控线程
performance_thread = threading.Thread(target=update_performance_data, daemon=True)
performance_thread.start()



# 初始化粒子
domain_size = (0.034, 0.034)
positions, velocities, radii, mass = initialize_particles(N=10000, domain_size=domain_size)
# 记录初始位置用于位移计算
initial_positions = positions.copy()

# 初始化ARF物理公式计算
print("正在初始化ARF物理公式计算...")
try:
    print("ARF物理公式计算初始化完成")
    print(f"声场参数: 频率={frequency}Hz, 振幅={get_amplitude()*1000:.3f}mm")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
except Exception as e:
    print(f"ARF物理公式初始化失败: {e}")
    print("将无法进行粒子力预测")

# 移除测试粒子相关代码，专注于ARF物理公式计算

if RUN_HEADLESS_BENCHMARK:
    # 仅运行计算，不进行任何渲染或动画，累计指定步数后退出并打印耗时
    steps = 40000
    dt = 1e-6
    print(f"Headless benchmark started: steps={steps}, dt={dt}")
    print("开始计算...")
    
    # 初始化7分箱设置（与主程序一致）
    comsol_edges_bench = np.array([-14.875, -10.625, -6.375, -2.125, 2.125, 6.375, 10.625, 14.875])
    edges_bench = comsol_edges_bench + 17.0  # 平移到DEM坐标系
    num_bins_bench = len(edges_bench) - 1  # 7个分箱
    M_total_bench = int(positions.shape[0] * 0.875)  # 只统计87.5%的粒子
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    bench_t0 = time.perf_counter()
    
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time

        # 力学更新：计算各项力（不在子模块中更新状态）
        total_force = np.zeros_like(positions)
        if USE_ARF_FORCES:
            arf_force = compute_pressure_gradient_and_apply_arf(
                positions, radii, t
            )

            total_force += arf_force

        if USE_STOKES_DRAG:
            drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                         time=t, use_sound_velocity=True, 
                                         frequency=frequency, sound_pressure_level=sound_pressure_level)
            total_force += drag_force

        # 显式欧拉积分（统一在主循环中进行）
        accelerations = total_force / mass[:, None]
        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt

        # 在第100个时步打印详细信息
        if step == 99:  # 第100个时步（从0开始计数）
            print(f"\n=== 第100个时步详细信息 (t = {t:.6f} s) ===")
            
            # 打印所有粒子的当前速度
            print(f"所有粒子当前速度 (m/s):")
            for i in range(min(10, len(velocities))):  # 只显示前10个粒子
                print(f"  粒子 {i}: vx = {velocities[i, 0]:.6e}, vy = {velocities[i, 1]:.6e}")
            if len(velocities) > 10:
                print(f"  ... (还有 {len(velocities) - 10} 个粒子)")
            
            # 计算并打印气流速度
            from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
            air_velocity = compute_air_velocity_due_to_sound(
                positions, t, frequency, sound_pressure_level
            )
            print(f"\n气流速度 (m/s):")
            for i in range(min(10, len(air_velocity))):  # 只显示前10个粒子位置的气流速度
                print(f"  位置 {i}: ux = {air_velocity[i, 0]:.6e}, uy = {air_velocity[i, 1]:.6e}")
            if len(air_velocity) > 10:
                print(f"  ... (还有 {len(air_velocity) - 10} 个位置)")
            
            # 打印压力梯度力（ARF力）
            if USE_ARF_FORCES:
                arf_force = compute_pressure_gradient_and_apply_arf(positions, radii, t)
                print(f"\n压力梯度力/ARF力 (N):")
                for i in range(min(10, len(arf_force))):  # 只显示前10个粒子
                    print(f"  粒子 {i}: Fx = {arf_force[i, 0]:.6e}, Fy = {arf_force[i, 1]:.6e}")
                if len(arf_force) > 10:
                    print(f"  ... (还有 {len(arf_force) - 10} 个粒子)")
            else:
                print(f"\n压力梯度力/ARF力: 未启用")
            
            # 打印斯托克斯阻力
            if USE_STOKES_DRAG:
                drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                             time=t, use_sound_velocity=True, 
                                             frequency=frequency, sound_pressure_level=sound_pressure_level)
                print(f"\n斯托克斯阻力 (N):")
                for i in range(min(10, len(drag_force))):  # 只显示前10个粒子
                    print(f"  粒子 {i}: Fx = {drag_force[i, 0]:.6e}, Fy = {drag_force[i, 1]:.6e}")
                if len(drag_force) > 10:
                    print(f"  ... (还有 {len(drag_force) - 10} 个粒子)")
            else:
                print(f"\n斯托克斯阻力: 未启用")
            
            # 打印总力
            print(f"\n总力 (N):")
            for i in range(min(10, len(total_force))):  # 只显示前10个粒子
                print(f"  粒子 {i}: Fx = {total_force[i, 0]:.6e}, Fy = {total_force[i, 1]:.6e}")
            if len(total_force) > 10:
                print(f"  ... (还有 {len(total_force) - 10} 个粒子)")
            
            # 打印相对速度（粒子速度 - 气流速度）
            relative_velocity = velocities - air_velocity
            print(f"\n相对速度 (粒子速度 - 气流速度) (m/s):")
            for i in range(min(10, len(relative_velocity))):  # 只显示前10个粒子
                print(f"  粒子 {i}: vx_rel = {relative_velocity[i, 0]:.6e}, vy_rel = {relative_velocity[i, 1]:.6e}")
            if len(relative_velocity) > 10:
                print(f"  ... (还有 {len(relative_velocity) - 10} 个粒子)")
            
            print(f"=== 第100个时步详细信息结束 ===\n")

        # 每10%进度打印一次
        progress_interval = max(1, steps // 10)  # 计算10%的步数间隔
        if (step + 1) % progress_interval == 0:
            current_time = time.perf_counter()
            elapsed_so_far = current_time - total_start_time
            progress = (step + 1) / steps * 100
            print(f"进度: {step + 1}/{steps} ({progress:.1f}%) - 已用时: {elapsed_so_far:.2f}s")
        if step == steps-1:
            # 记录最终粒子位置用于分箱统计
            final_positions = positions.copy()
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    bench_t1 = time.perf_counter()
    physics_elapsed = bench_t1 - bench_t0
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"物理计算时间: {physics_elapsed:.4f} s")
    print(f"最终仿真时间: {simulation_time:.6f} s")


    # 计算完成后，绘制一次静态图（粒子位置 + 密度分布），不保存，仅展示
    def _optimize_bandwidth_1d(x_coords_mm, max_samples_for_cv=50):
        """优化KDE带宽因子（相对Scott规则的乘数，越小越窄）。"""
        x = np.asarray(x_coords_mm)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return None
        # 目标函数：负对数似然（采样最多max_samples_for_cv个点加速）
        idx = np.arange(x.size)
        if x.size > max_samples_for_cv:
            rng = np.random.default_rng(42)
            cv_idx = rng.choice(idx, size=max_samples_for_cv, replace=False)
        else:
            cv_idx = idx

        def objective(factor):
            if factor <= 0:
                return np.inf
            log_like = 0.0
            for i in cv_idx:
                train = np.delete(x, i)
                try:
                    kde = gaussian_kde(train, bw_method=float(factor))
                    pdf_val = float(kde(x[i])[0])
                    if pdf_val > 0:
                        log_like += np.log(pdf_val)
                except Exception:
                    return np.inf
            return -log_like

        # 搜索较小因子以保留峰值，同时避免>1导致过度平滑
        res = minimize_scalar(objective, bounds=(0.05, 1.0), method='bounded')
        return float(res.x) if res.success else 1.0

    def _kde_density_1d(x_coords_mm, eval_grid_mm, bw=None):
        x = np.asarray(x_coords_mm)
        if x.size == 0:
            return np.zeros_like(eval_grid_mm)
        if bw is None:
            bw = _optimize_bandwidth_1d(x)
        kde = gaussian_kde(x, bw_method=float(bw))
        return kde(eval_grid_mm)

    def _smooth_preserve_minmax(_y, window_ratio_base=0.07, window_ratio_strong=0.15, ptp_flat_threshold=0.5, min_limit=-99.9):
        """自适应Hanning平滑：
        - 变化小（峰-谷幅度 < 阈值）时：更强平滑且不做极值重标定，避免引入虚假正弦；
        - 变化明显时：常规平滑并线性重标定回原始[min, max]以保持峰谷；
        - 始终进行反射填充，曲线无断点；并裁切至接近-100%。
        参数单位：_y为百分比数据（%），ptp_flat_threshold为百分比点阈值。
        """
        y = np.asarray(_y)
        n = y.size
        if n < 5:
            return np.maximum(y, min_limit)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        ptp = y_max - y_min

        # 根据幅度选择窗口与是否重标定
        is_flat = bool(ptp <= float(ptp_flat_threshold))
        window_ratio = float(window_ratio_strong if is_flat else window_ratio_base)

        w = max(5, int(n * window_ratio))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        if w < 5:
            return np.maximum(y, min_limit)

        pad = w // 2
        y_pad = np.pad(y, (pad, pad), mode='reflect')
        kernel = np.hanning(w)
        kernel = kernel / np.sum(kernel)
        y_s = np.convolve(y_pad, kernel, mode='valid')

        if not is_flat:
            ys_min = float(np.min(y_s))
            ys_max = float(np.max(y_s))
            if ys_max > ys_min and y_max > y_min:
                a = (y_max - y_min) / (ys_max - ys_min)
                b = y_min - a * ys_min
                y_s = a * y_s + b

        y_s = np.maximum(y_s, min_limit)
        return y_s

    fig_h, (ax_p_h, ax_d_h) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：粒子散点（单位mm）
    ax_p_h.set_xlim(0, domain_size[0] * 1000)
    ax_p_h.set_ylim(0, domain_size[1] * 1000)
    ax_p_h.set_title("Particle Positions (after computation)")
    ax_p_h.set_xlabel("x (mm)")
    ax_p_h.set_ylabel("y (mm)")
    ax_p_h.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, s=(radii * 1e6 * 4)**2, c='blue', alpha=0.6)

    # 右图：基于距离的浓度分布变化率（柱状图）
    # 使用与主程序相同的浓度计算机制
    x_grid_h = np.linspace(0.0, 34.0, 100)  # 100个点用于平滑显示
    
    def _calculate_concentration_distribution_h(positions):
        """基于粒子到参考位置的距离计算浓度分布 - 直接统计法，避免削峰"""
        try:
            # 获取当前帧的粒子位置（单位：mm）
            current_positions = positions[:, 0] * 1000.0  # 只取x坐标
            current_positions = current_positions[np.isfinite(current_positions)]
            
            if len(current_positions) == 0:
                return np.zeros_like(x_grid_h)
            
            # 计算每个x_grid位置处的浓度
            concentrations = np.zeros_like(x_grid_h)
            
            # 定义统计窗口大小（mm）
            window_size = 0.2  # 0.5mm的统计窗口
            
            for i, x_pos in enumerate(x_grid_h):
                # 计算在统计窗口内的粒子数量
                # 使用简单的矩形窗口，避免高斯平滑
                in_window = np.abs(current_positions - x_pos) <= window_size
                concentration = np.sum(in_window)
                
                concentrations[i] = concentration
            
            return concentrations
        except Exception as e:
            print(f"浓度计算错误: {e}")
            return np.zeros_like(x_grid_h)
    
    # 计算初始和最终浓度分布
    initial_concentrations_h = _calculate_concentration_distribution_h(initial_positions)
    final_concentrations_h = _calculate_concentration_distribution_h(final_positions)
    
    # 计算相对变化
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change_h = (final_concentrations_h - initial_concentrations_h) / initial_concentrations_h * 100.0
        relative_change_h = np.nan_to_num(relative_change_h, nan=0.0, posinf=0.0, neginf=0.0)
    
    bars_h = ax_d_h.bar(x_grid_h, relative_change_h, width=0.34, align='center', alpha=0.8)
    # 平滑连接柱顶的曲线（PCHIP）
    try:
        pchip_h = PchipInterpolator(x_grid_h, relative_change_h, extrapolate=False)
        x_smooth_h = np.linspace(x_grid_h[0], x_grid_h[-1], max(50, 4 * len(x_grid_h)))
        y_smooth_h = pchip_h(x_smooth_h)
        curve_line_h, = ax_d_h.plot(x_smooth_h, y_smooth_h, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line_h = None
    ax_d_h.set_xlim(x_grid_h[0], x_grid_h[-1])
    if np.any(np.isfinite(relative_change_h)):
        y_min_h = float(np.nanmin(relative_change_h))
        y_max_h = float(np.nanmax(relative_change_h))
        y_margin_h = (y_max_h - y_min_h) * 0.1 if y_max_h > y_min_h else 1.0
        ax_d_h.set_ylim(y_min_h - y_margin_h, y_max_h + y_margin_h)
    ax_d_h.set_title("Particle Density Change Rate (after computation)")
    ax_d_h.set_xlabel("x (mm)")
    ax_d_h.set_ylabel("Concentration Change (%)")
    ax_d_h.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 保证在绘制右图后插入：
    np.savetxt('density_curve_DEM.txt', np.vstack([x_grid_h, relative_change_h]).T)

    sys.exit(0)

# 绘图初始化 - 创建两个子图
fig, (ax_particles, ax_density) = plt.subplots(1, 2, figsize=(16, 6))

# 左侧子图：粒子运动
ax_particles.set_xlim(0, domain_size[0] * 1000)
ax_particles.set_ylim(0, domain_size[1] * 1000)
ax_particles.set_title("Particle Motion")
ax_particles.set_xlabel("x (mm)")
ax_particles.set_ylabel("y (mm)")

# 右侧子图：密度变化曲线
ax_density.set_xlim(0.0, 34.0)  # 更新为与x_grid一致的范围
ax_density.set_ylim(-5, 5)  # 进一步缩小y轴范围，确保曲线可见
ax_density.set_title("Particle Density Distribution")
ax_density.set_xlabel("x (mm)")
ax_density.set_ylabel("Concentration Change (%)")
ax_density.grid(True, alpha=0.3)

# 移除测试粒子图例

# 声场
X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=0.0)
sound_img = ax_particles.imshow(
    P,
    extent=(0, domain_size[0] * 1000, 0, domain_size[1] * 1000),
    origin='lower',
    cmap='RdBu_r',
    alpha=0.4,
    vmin=-get_amplitude(),
    vmax=get_amplitude()
)

# 普通粒子
scatter = ax_particles.scatter(positions[:, 0] * 1000,
                     positions[:, 1] * 1000,
                     s=(radii * 1e6 * 4)**2,
                     c='blue', alpha=0.6)

# 移除测试粒子和尾流箭头相关代码

# 添加性能信息文本框
performance_text = ax_particles.text(0.02, 0.98, '', transform=ax_particles.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

"""初始化基于距离的浓度分布机制（来自comsol_visualizer.py）"""
# 创建x轴网格用于显示浓度分布
x_grid = np.linspace(0.0, 34.0, 100)  # 100个点用于平滑显示
concentration_values = np.zeros_like(x_grid)

# 浓度统计参数
concentration_scale = 100.0  # 浓度缩放因子
distance_weight = 1.0  # 距离权重
baseline_concentration = 0.0  # 基准浓度（相对变化）

# 计算初始浓度分布
def _calculate_concentration_distribution(positions):
    """基于粒子到参考位置的距离计算浓度分布 - 直接统计法，避免削峰"""
    try:
        # 获取当前帧的粒子位置（单位：mm）
        current_positions = positions[:, 0] * 1000.0  # 只取x坐标
        current_positions = current_positions[np.isfinite(current_positions)]
        
        if len(current_positions) == 0:
            return np.zeros_like(x_grid)
        
        # 计算每个x_grid位置处的浓度
        concentrations = np.zeros_like(x_grid)
        
        # 定义统计窗口大小（mm）
        window_size = 0.5  # 0.5mm的统计窗口
        
        for i, x_pos in enumerate(x_grid):
            # 计算在统计窗口内的粒子数量
            # 使用简单的矩形窗口，避免高斯平滑
            in_window = np.abs(current_positions - x_pos) <= window_size
            concentration = np.sum(in_window)
            
            concentrations[i] = concentration
        
        return concentrations
    except Exception as e:
        print(f"浓度计算错误: {e}")
        return np.zeros_like(x_grid)

# 计算初始浓度分布
initial_concentrations = _calculate_concentration_distribution(positions)
baseline_concentration = np.mean(initial_concentrations)

# 调试信息：打印浓度统计设置
print(f"\n[调试] 基于距离的浓度统计设置:")
print(f"  x_grid范围: {x_grid[0]:.1f}mm - {x_grid[-1]:.1f}mm")
print(f"  浓度缩放因子: {concentration_scale}")
print(f"  距离权重: {distance_weight}")
print(f"  基准浓度: {baseline_concentration:.3f}")
print(f"  总粒子数: {len(positions)}")

# 计算初始相对变化
with np.errstate(divide='ignore', invalid='ignore'):
    relative_change0 = (initial_concentrations - initial_concentrations) / initial_concentrations * 100.0
    relative_change0 = np.nan_to_num(relative_change0, nan=0.0, posinf=0.0, neginf=0.0)

# 初始化柱状图（使用x_grid作为显示）
bars = ax_density.bar(x_grid, relative_change0, width=0.34, align='center', alpha=0.8, label='Concentration Change')
# 初始平滑曲线
try:
    pchip0 = PchipInterpolator(x_grid, relative_change0, extrapolate=False)
    x_smooth0 = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
    y_smooth0 = pchip0(x_smooth0)
    curve_line, = ax_density.plot(x_smooth0, y_smooth0, 'r-', linewidth=2, label='Smoothed')
except Exception:
    curve_line = ax_density.plot([], [], 'r-', linewidth=2, label='Smoothed')[0]
ax_density.set_xlim(x_grid[0], x_grid[-1])
if np.any(np.isfinite(relative_change0)):
    y_min0 = float(np.nanmin(relative_change0))
    y_max0 = float(np.nanmax(relative_change0))
    y_margin0 = (y_max0 - y_min0) * 0.1 if y_max0 > y_min0 else 1.0
    ax_density.set_ylim(y_min0 - y_margin0, y_max0 + y_margin0)
ax_density.legend()



# 帧率计算变量
frame_times = []
last_frame_time = time.time()

def update(frame):
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, step_timing_printed

    # 添加调试信息
    if frame < 5:  # 只在前5帧打印调试信息
        print(f"Frame {frame}: Starting update function")

    # 渲染间隔（仅控制可视化与UI的更新频率；物理仍每帧推进）
    render_interval = 1  # 每帧都渲染，确保动画流畅
    should_render = (frame % render_interval == 0)

    # 开始计时（第5帧时）
    if frame == 5 and not step_timing_printed:
        t0 = time.perf_counter()

    # 计算帧率
    current_time = time.time()
    frame_time = current_time - last_frame_time
    frame_times.append(frame_time)
    
    # 保持最近30帧的时间记录
    if len(frame_times) > 30:
        frame_times.pop(0)
    
    # 计算平均帧率
    if len(frame_times) > 0:
        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        fps = 0
    
    performance_data['fps'] = fps
    performance_data['frame_time'] = frame_time * 1000  # 转换为毫秒
    
    last_frame_time = current_time

    dt = 1e-8
    global simulation_time
    simulation_time += dt  # 累计仿真时间
    t = simulation_time  # 使用累计时间

    # 计时：声场计算开始
    if frame == 5 and not step_timing_printed:
        t1 = time.perf_counter()

    # 使用ARF与阻力计算力，并在此处做积分
    total_force = np.zeros_like(positions)
    if USE_ARF_FORCES:
        if frame < 5:
            print(f"Frame {frame}: Computing ARF forces")
        try:
            arf_force = compute_pressure_gradient_and_apply_arf(
                positions, radii, t
            )
            print(t)
            total_force += arf_force
            if frame < 5:
                print(f"Frame {frame}: ARF forces computed successfully")
        except Exception as e:
            print(f"Frame {frame}: ARF calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 计时：ARF计算结束，斯托克斯阻力开始
    if frame == 5 and not step_timing_printed:
        t2 = time.perf_counter()
    
    # 应用斯托克斯阻力（力）
    if USE_STOKES_DRAG:
        drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                     time=t, use_sound_velocity=True, 
                                     frequency=frequency, sound_pressure_level=sound_pressure_level)
        total_force += drag_force

    # 统一积分更新
    accelerations = total_force / mass[:, None]
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt
    
    # 计时：斯托克斯阻力结束，可视化更新开始
    if frame == 5 and not step_timing_printed:
        t3 = time.perf_counter()

    # 更新声场 & 粒子位置（仅在需要渲染的帧进行可视化更新）
    if should_render:
        Nx, Ny = (200, 200)
        _, _, P = compute_sound_field(domain_size=domain_size, resolution=(Nx, Ny), time=t)
        sound_img.set_data(P)
        scatter.set_offsets(positions * 1000)
    
    # 计时：声场和散点图更新结束，密度计算开始
    if frame == 5 and not step_timing_printed:
        t4 = time.perf_counter()
    
    if should_render:
        # 更新基于距离的浓度分布
        current_concentrations = _calculate_concentration_distribution(positions)
        
        # 计算相对变化
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (current_concentrations - initial_concentrations) / initial_concentrations * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新每个bar的高度
        for rect, h in zip(bars, relative_change):
            rect.set_height(h)

        # 更新平滑曲线（PCHIP 连接柱顶）
        try:
            pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
            x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
            y_smooth = pchip(x_smooth)
            curve_line.set_data(x_smooth, y_smooth)
        except Exception:
            curve_line.set_data(x_grid, relative_change)

        # 动态调整y轴范围
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax_density.set_ylim(y_min - y_margin, y_max + y_margin)

        ax_density.set_title(f"Particle Density Change Rate (t = {t:.6f} s)")
    
    # 计时：密度计算结束，性能显示开始
    if frame == 5 and not step_timing_printed:
        t5 = time.perf_counter()


    
    # 每0.01s保存一次密度分布图，从0.01s到0.2s（仅在渲染帧进行保存与绘制）
    if should_render:
        if t >= 0.01 and t <= 0.2 and (t - last_save_time) >= 0.01:
            # 创建密度分布图
            plt.figure(figsize=(12, 8))
            plt.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, label='Concentration Change (%)')
            plt.xlabel('x (mm)')
            plt.ylabel('Concentration Change (%)')
            plt.title(f'Particle Density Change Rate at t = {t:.6f} s')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存图片，文件名包含时间信息
            filename = f'density_plot_{int(t*1000):03d}_t_{t:.6f}s.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            last_save_time = t  # 更新上次保存时间

    # 更新性能信息显示
    arf_status = "✓ ARF Active" if USE_ARF_FORCES else "✗ ARF Inactive"
    stokes_status = "✓ Stokes Drag Active" if USE_STOKES_DRAG else "✗ Stokes Drag Inactive"
    if should_render:
        performance_text.set_text(
            f'Simulation Time: {t:.6f} s\n'
            f'FPS: {performance_data["fps"]:.1f}\n'
            f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
            f'Particles: {performance_data["particle_count"]}\n'
            f'ARF Physics: {arf_status}\n'
            f'Stokes Drag: {stokes_status}'
        )
    
    # 计时：性能显示结束
    if frame == 5 and not step_timing_printed:
        t6 = time.perf_counter()
    
    # 触发一次立即绘制并测量（仅在第5帧做一次）
    if frame == 5 and not step_timing_printed:
        pending_draw_measurement = {'draw_start': time.perf_counter(), 'draw_end': None}
        fig.canvas.draw()
        if pending_draw_measurement['draw_end'] is None:
            # 后备：立刻取一次时间，包含渲染阻塞
            pending_draw_measurement['draw_end'] = time.perf_counter()

        def ms(x):
            return max(0.0, x) * 1000.0

        # 计算各阶段耗时（缺失时间戳用相邻前一刻兜底，避免未定义）
        t1_l = locals().get('t1', t0)
        t2_l = locals().get('t2', t1_l)
        t3_l = locals().get('t3', t2_l)
        t4_l = locals().get('t4', t3_l)
        t5_l = locals().get('t5', t4_l)
        t6_l = locals().get('t6', t5_l)

        t_sound = ms(t1_l - t0)
        t_arf = ms(t2_l - t1_l)
        t_drag = ms(t3_l - t2_l)
        t_scatter = ms(t4_l - t3_l)
        t_density = ms(t5_l - t4_l)
        t_perf = ms(t6_l - t5_l)
        draw_ms = ms(pending_draw_measurement['draw_end'] - pending_draw_measurement['draw_start'])
        total_ms = t_sound + t_arf + t_drag + t_scatter + t_density + t_perf + draw_ms

        print("\nStep timing breakdown (frame 5):")
        print(f"  Particles:          {len(positions)}")
        print(f"  Sound field update: {t_sound:.2f} ms")
        print(f"  ARF computation:    {t_arf:.2f} ms")
        print(f"  Stokes drag:        {t_drag:.2f} ms")
        print(f"  Scatter update:     {t_scatter:.2f} ms")
        print(f"  Density calculation:{t_density:.2f} ms")
        print(f"  Perf display:       {t_perf:.2f} ms")
        print(f"  Render (draw):      {draw_ms:.2f} ms")
        print(f"  Total (incl. draw): {total_ms:.2f} ms")
        step_timing_printed = True
    
    return sound_img, scatter, performance_text, bars


ani = animation.FuncAnimation(fig, update, frames=20000, interval=50, blit=False)  # 50ms间隔，约20fps
plt.show()
