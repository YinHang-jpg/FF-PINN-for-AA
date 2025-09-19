import matplotlib.pyplot as plt
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

# 配置开关 - 使用ARF物理公式进行力预测，同时开启斯托克斯阻力
RUN_HEADLESS_BENCHMARK = True  # 关闭渲染与动画，仅计算并在1000步后退出
USE_TRAVELING_WAVE = False
USE_TEST_PARTICLE = False
USE_ARF_FORCES = True  # 使用ARF物理公式计算所有粒子受力
USE_GRAVITY = False
USE_STOKES_DRAG = False  # 开启斯托克斯阻力
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
    from initialization.sound_source_standing import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = True

# 导入ARF物理公式计算模块
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf

# 导入斯托克斯阻力模块
from mechanisms.Stokes_drag import apply_stokes_drag

# 仅用于打印：根据当前速度计算每个粒子的斯托克斯阻力（与 mechanisms/Stokes_drag.py 一致）
def _cunningham_correction_factor(d_p, lambda_g):
    ratio = d_p / lambda_g
    exp_term = np.exp(-0.550 * ratio)
    bracket_term = 2.514 + 0.800 * exp_term
    return 1.0 + bracket_term * ratio

def compute_stokes_drag_force(positions, velocities, radii, viscosity=1.79e-5, fluid_velocity=None, lambda_g=6.5e-8):
    N = len(positions)
    if fluid_velocity is None:
        fluid_velocity = np.zeros((N, 2))
    drag_force = np.zeros_like(positions)
    for i in range(N):
        d_p = 2.0 * radii[i]
        C_c = _cunningham_correction_factor(d_p, lambda_g)
        relative_velocity = velocities[i] - fluid_velocity[i]
        drag_coeff = 3.0 * np.pi * viscosity * d_p / C_c
        drag_force[i] = -drag_coeff * relative_velocity
    return drag_force

def compute_particle_forces_using_arf(positions, velocities, mass, radii, dt, time, print_forces=False):
    """
    使用ARF物理公式计算粒子受力
    - print_forces: 是否打印力信息
    """
    try:
        # 使用ARF物理公式计算声辐射力
        positions, velocities = compute_pressure_gradient_and_apply_arf(
            positions, velocities, mass, radii, dt, domain_size, (200, 200), time, compute_sound_field, IS_STANDING_WAVE
        )
        
        # 计算力信息用于打印（如果需要）
        if print_forces:
            # 重新计算力用于显示（这里简化处理，直接使用ARF模块中的计算）
            from mechanisms.ARF import get_particle_arf_force
            arf_forces = np.zeros_like(positions)
            for i in range(len(positions)):
                arf_forces[i] = get_particle_arf_force(positions[i, 0], positions[i, 1], radii[i], time)
            
            stokes_force = compute_stokes_drag_force(positions, velocities, radii) if USE_STOKES_DRAG else np.zeros_like(positions)
            print(f"\n=== 时间步 {time:.6f}s 的粒子受力 ===")
            print("粒子ID | X位置(mm) | Y位置(mm) | ARF Fx(pN) | ARF Fy(pN) | Stokes Fx(pN) | Stokes Fy(pN)")
            print("-" * 110)
            for i in range(len(positions)):
                print(f"{i:6d} | {positions[i,0]*1000:8.3f} | {positions[i,1]*1000:8.3f} | {arf_forces[i,0]*1e12:12.3e} | {arf_forces[i,1]*1e12:12.3e} | {stokes_force[i,0]*1e12:13.3e} | {stokes_force[i,1]*1e12:13.3e}")
            print(f"总粒子数: {len(positions)}")
            print(f"ARF 平均Fx: {np.mean(arf_forces[:,0])*1e12:.3e} pN | 范围: [{np.min(arf_forces[:,0])*1e12:.3e}, {np.max(arf_forces[:,0])*1e12:.3e}] pN")
            if USE_STOKES_DRAG:
                print(f"Stokes 平均Fx: {np.mean(stokes_force[:,0])*1e12:.3e} pN | 范围: [{np.min(stokes_force[:,0])*1e12:.3e}, {np.max(stokes_force[:,0])*1e12:.3e}] pN")
            print("=" * 110)
        
        return positions, velocities, None
            
    except Exception as e:
        print(f"ARF物理公式计算力时出错: {e}")
        print("将使用零力（无外力效应）")
        # 如果ARF计算失败，不施加任何力，保持原有运动
        return positions, velocities, None
# 移除所有力学计算模块的导入，现在只使用ARF物理公式

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
    print(f"声场参数: 频率={frequency}Hz, 振幅={amplitude*1000:.3f}mm")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
except Exception as e:
    print(f"ARF物理公式初始化失败: {e}")
    print("将无法进行粒子力预测")

# 移除测试粒子相关代码，专注于ARF物理公式计算

if RUN_HEADLESS_BENCHMARK:
    # 仅运行计算，不进行任何渲染或动画，累计指定步数后退出并打印耗时
    steps = 1000
    dt = 1e-7
    print(f"Headless benchmark started: steps={steps}, dt={dt}")
    print("开始计算...")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    bench_t0 = time.perf_counter()
    
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time

        # 力学更新：ARF物理公式
        if USE_ARF_FORCES:
            positions, velocities, _ = compute_particle_forces_using_arf(
                positions, velocities, mass, radii, dt, t, print_forces=False
            )

        # 斯托克斯阻力
        if USE_STOKES_DRAG:
            positions, velocities = apply_stokes_drag(positions, velocities, radii, mass, dt)
        
        # 每10000步打印一次进度
        if (step + 1) % 10000 == 0:
            current_time = time.perf_counter()
            elapsed_so_far = current_time - total_start_time
            progress = (step + 1) / steps * 100
            print(f"进度: {step + 1}/{steps} ({progress:.1f}%) - 已用时: {elapsed_so_far:.2f}s")

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
    def _compute_density(_x_positions_mm, _x_grid_mm, bandwidth_mm=1.5):
        d = np.zeros_like(_x_grid_mm)
        for _x in _x_positions_mm:
            d += np.exp(-0.5 * ((_x_grid_mm - _x) / bandwidth_mm) ** 2)
        return d

    fig_h, (ax_p_h, ax_d_h) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：粒子散点（单位mm）
    ax_p_h.set_xlim(0, domain_size[0] * 1000)
    ax_p_h.set_ylim(0, domain_size[1] * 1000)
    ax_p_h.set_title("Particle Positions (after computation)")
    ax_p_h.set_xlabel("x (mm)")
    ax_p_h.set_ylabel("y (mm)")
    ax_p_h.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, s=(radii * 1e6 * 4)**2, c='blue', alpha=0.6)

    # 右图：密度分布（相对变化百分比，与动画一致）
    x_grid_h = np.linspace(0.0, domain_size[0] * 1000.0, 200)
    initial_x_mm_h = initial_positions[:, 0] * 1000.0
    current_x_mm = positions[:, 0] * 1000.0
    initial_density_h = _compute_density(initial_x_mm_h, x_grid_h, bandwidth_mm=1.5)
    baseline_h = np.mean(initial_density_h)
    current_density_h = _compute_density(current_x_mm, x_grid_h, bandwidth_mm=1.5)
    relative_change_h = (current_density_h - baseline_h) / baseline_h * 100.0
    ax_d_h.plot(x_grid_h, relative_change_h, 'b-', linewidth=2, label='Relative Change (%)')
    # 动态y轴范围（10%边距）
    y_min_h = float(np.min(relative_change_h))
    y_max_h = float(np.max(relative_change_h))
    y_margin_h = (y_max_h - y_min_h) * 0.1 if y_max_h > y_min_h else 1.0
    ax_d_h.set_ylim(y_min_h - y_margin_h, y_max_h + y_margin_h)
    ax_d_h.set_title("Particle Density Distribution (after computation)")
    ax_d_h.set_xlabel("x (mm)")
    ax_d_h.set_ylabel("Relative Change (%)")
    ax_d_h.grid(True, alpha=0.3)
    ax_d_h.legend()

    plt.tight_layout()
    plt.show()

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
ax_density.set_xlim(0, domain_size[0] * 1000)
ax_density.set_ylim(-5, 5)  # 进一步缩小y轴范围，确保曲线可见
ax_density.set_title("Particle Density Distribution")
ax_density.set_xlabel("x (mm)")
ax_density.set_ylabel("Relative Change (%)")
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
    vmin=-amplitude,
    vmax=amplitude
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

# 初始化密度分布曲线（使用核密度估计获得平滑曲线）
# 创建x坐标网格（单位：mm）
x_grid = np.linspace(0.0, domain_size[0] * 1000.0, 200)
# 计算初始粒子的x坐标（单位：mm）
initial_x_positions = positions[:, 0] * 1000.0

# 使用高斯核密度估计计算初始密度分布
def compute_density(x_positions, x_grid, bandwidth=1.0):
    """使用高斯核计算密度分布"""
    density = np.zeros_like(x_grid)
    for x in x_positions:
        # 高斯核：exp(-(x-x_i)^2 / (2*bandwidth^2))
        density += np.exp(-0.5 * ((x_grid - x) / bandwidth) ** 2)
    return density

# 计算初始密度分布
initial_density = compute_density(initial_x_positions, x_grid, bandwidth=1.5)
# 归一化到相对变化百分比
baseline = np.mean(initial_density)
initial_relative = (initial_density - baseline) / baseline * 100.0

# 绘制初始密度曲线
density_line, = ax_density.plot(x_grid, initial_relative, 'b-', linewidth=2, label='Relative change')
ax_density.legend()



# 帧率计算变量
frame_times = []
last_frame_time = time.time()

def update(frame):
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, step_timing_printed

    # 渲染间隔（仅控制可视化与UI的更新频率；物理仍每帧推进）
    render_interval = 1000
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

    dt = 2.5e-6
    global simulation_time
    simulation_time += dt  # 累计仿真时间
    t = simulation_time  # 使用累计时间

    # 计时：声场计算开始
    if frame == 5 and not step_timing_printed:
        t1 = time.perf_counter()

    # 使用ARF物理公式计算粒子受力
    if USE_ARF_FORCES:
        positions, velocities, forces = compute_particle_forces_using_arf(
            positions, velocities, mass, radii, dt, t, print_forces=False
        )
    
    # 计时：ARF计算结束，斯托克斯阻力开始
    if frame == 5 and not step_timing_printed:
        t2 = time.perf_counter()
    
    # 应用斯托克斯阻力
    if USE_STOKES_DRAG:
        positions, velocities = apply_stokes_drag(positions, velocities, radii, mass, dt)
    
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
        # 更新密度分布曲线（使用核密度估计）
        current_x_positions = positions[:, 0] * 1000.0
        current_density = compute_density(current_x_positions, x_grid, bandwidth=1.5)
        relative_change = (current_density - baseline) / baseline * 100.0
        density_line.set_data(x_grid, relative_change)

        # 动态调整y轴范围，确保曲线完整显示
        y_min = np.min(relative_change)
        y_max = np.max(relative_change)
        y_margin = (y_max - y_min) * 0.1  # 添加10%的边距
        ax_density.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax_density.set_title(f"Particle Density Distribution (t = {t:.6f} s)")
    
    # 计时：密度计算结束，性能显示开始
    if frame == 5 and not step_timing_printed:
        t5 = time.perf_counter()


    
    # 每0.01s保存一次密度分布图，从0.01s到0.2s（仅在渲染帧进行保存与绘制）
    if should_render:
        if t >= 0.01 and t <= 0.2 and (t - last_save_time) >= 0.01:
            # 创建密度分布图
            plt.figure(figsize=(12, 8))
            plt.plot(x_grid, relative_change, 'b-', linewidth=2, label='Relative Change (%)')
            plt.xlabel('x (mm)')
            plt.ylabel('Relative Change (%)')
            plt.title(f'Particle Density Distribution at t = {t:.6f} s')
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
    
    return sound_img, scatter, performance_text, density_line


ani = animation.FuncAnimation(fig, update, frames=20000, interval=0, blit=False)
plt.show()
