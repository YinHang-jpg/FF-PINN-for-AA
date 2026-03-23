import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import torch
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import compute_sound_field as compute_standing_wave, amplitude as amplitude_standing
from initialization.sound_source_traveling import compute_sound_field as compute_traveling_wave, amplitude as amplitude_traveling
from PINN.ARF_PINN import PINNARF, train_pinn_arf_model
from mechanisms.Stokes_drag import apply_stokes_drag

# 配置开关
USE_TRAVELING_WAVE = False
TRAIN_MODEL = False  # 是否训练新模型
USE_PRETRAINED_MODEL = True  # 是否使用预训练模型
USE_GPU = True  # 是否使用GPU加速
BATCH_SIZE = 1000  # GPU批处理大小（增加批处理大小以提升性能）
PERFORMANCE_TEST = False  # 是否运行性能对比测试

# GPU设备检测和配置
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"[GPU] GPU加速已启用: {torch.cuda.get_device_name(0)}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
elif USE_GPU:
    print("[WARNING] GPU不可用，使用CPU")
    device = torch.device('cpu')
    USE_GPU = False
else:
    device = torch.device('cpu')
    print("使用CPU计算")

# 声场类型选择
if USE_TRAVELING_WAVE:
    compute_sound_field = compute_traveling_wave
    amplitude = amplitude_traveling
    IS_STANDING_WAVE = False
    print("Using traveling wave sound field")
else:
    compute_sound_field = compute_standing_wave
    amplitude = amplitude_standing
    IS_STANDING_WAVE = True
    print("Using standing wave sound field")

# 初始化PINN ARF模型
if TRAIN_MODEL:
    print("Training new PINN ARF model...")
    train_pinn_arf_model("PINN/arf_model.pth")

if USE_PRETRAINED_MODEL:
    try:
        pinn_arf = PINNARF("PINN/arf_model.pth")
        print("✅ Pre-trained PINN model loaded successfully")
    except:
        print("⚠️ Failed to load pre-trained model, using theoretical calculation")
        pinn_arf = PINNARF()
else:
    pinn_arf = PINNARF()

# 初始化粒子
domain_size = (0.01, 0.01)
N_particles = 50000  # 与DEM_main.py保持一致
positions, velocities, radii, mass = initialize_particles(N=N_particles, domain_size=domain_size)

# 粒子直径（用于ARF计算）
particle_diameters = np.ones(N_particles) * 1e-6

# 将粒子数据转换为GPU张量（如果使用GPU）
if USE_GPU:
    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    velocities = torch.tensor(velocities, dtype=torch.float32, device=device)
    radii = torch.tensor(radii, dtype=torch.float32, device=device)
    mass = torch.tensor(mass, dtype=torch.float32, device=device)
    particle_diameters = torch.tensor(particle_diameters, dtype=torch.float32, device=device)
    print(f"[GPU] 粒子数据已移至GPU: {N_particles} 个粒子")
else:
    print(f"使用CPU处理: {N_particles} 个粒子")

# 绘图初始化
fig, ax = plt.subplots(figsize=(8, 6))  # 增加图形大小以容纳性能显示
ax.set_xlim(0, domain_size[0] * 1000)
ax.set_ylim(0, domain_size[1] * 1000)
ax.set_title(f"PINN Simulation - {'Traveling' if USE_TRAVELING_WAVE else 'Standing'} Wave")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

# 声场可视化
X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=0.0)
sound_img = ax.imshow(
    P,
    extent=(0, domain_size[0] * 1000, 0, domain_size[1] * 1000),
    origin='lower',
    cmap='RdBu_r',
    alpha=0.4,
    vmin=-amplitude,
    vmax=amplitude
)

# 添加颜色条
cbar = plt.colorbar(sound_img, ax=ax, shrink=0.8)
cbar.set_label('Sound Pressure (Pa)', fontsize=10)

# 粒子散点图
if USE_GPU:
    # GPU版本：需要转换为CPU张量
    scatter = ax.scatter(positions[:, 0].cpu().numpy() * 1000,
                         positions[:, 1].cpu().numpy() * 1000,
                         s=(radii.cpu().numpy() * 1e6 * 8)**2,
                         c='blue', alpha=0.6, label='Particles')
else:
    # CPU版本
    scatter = ax.scatter(positions[:, 0] * 1000,
                         positions[:, 1] * 1000,
                         s=(radii * 1e6 * 8)**2,
                         c='blue', alpha=0.6, label='Particles')

# 添加图例
ax.legend(loc='upper right')

# 性能统计
performance_stats = {
    'total_frames': 0,
    'total_time': 0.0,
    'arf_calls': 0,
    'last_frame_time': time.time(),
    'fps_history': [],
    'cpu_history': [],
    'memory_history': [],
    'memory_percent_history': [],
    'total_memory_gb': 8.0
}

def get_system_metrics():
    """获取系统性能指标"""
    try:
        # 尝试使用psutil获取详细指标
        import psutil
        current_process = psutil.Process()
        
        # CPU使用率需要多次调用才能获得准确值
        # 第一次调用通常返回0，需要等待一段时间
        cpu_percent = current_process.cpu_percent(interval=0.1)
        
        # 内存使用
        memory_info = current_process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # 获取系统总内存
        system_memory = psutil.virtual_memory()
        total_memory_gb = system_memory.total / (1024**3)
        memory_percent = current_process.memory_percent()
        
        return cpu_percent, memory_mb, memory_percent, total_memory_gb
    except ImportError:
        # 如果没有psutil，使用简化方法
        try:
            import os
            # 获取进程内存使用（仅Linux/macOS）
            if os.name == 'posix':
                import resource
                memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                memory_mb = memory_kb / 1024 if os.uname().sysname == 'Darwin' else memory_kb
                # 简化CPU估算（基于时间）
                cpu_percent = 50.0  # 默认值
                memory_percent = 5.0  # 默认值
                total_memory_gb = 8.0  # 默认值
                return cpu_percent, memory_mb, memory_percent, total_memory_gb
            else:
                # Windows或其他系统
                return 50.0, 100.0, 5.0, 8.0  # 默认值
        except:
            # 如果都失败，返回默认值
            return 50.0, 100.0, 5.0, 8.0  # 默认值

# 初始化时获取一次系统指标
try:
    import psutil
    current_process = psutil.Process()
    # 预热CPU测量
    current_process.cpu_percent(interval=0.1)
    initial_cpu, initial_memory, initial_memory_percent, total_memory_gb = get_system_metrics()
    performance_stats['cpu_history'].append(initial_cpu)
    performance_stats['memory_history'].append(initial_memory)
    performance_stats['memory_percent_history'].append(initial_memory_percent)
    performance_stats['total_memory_gb'] = total_memory_gb
    print(f"Initial system metrics - CPU: {initial_cpu:.1f}%, Memory: {initial_memory:.1f} MB ({initial_memory_percent:.1f}%), Total RAM: {total_memory_gb:.1f} GB")
except:
    print("Warning: Could not initialize system metrics monitoring")
    performance_stats['cpu_history'].append(0.0)
    performance_stats['memory_history'].append(100.0)
    performance_stats['memory_percent_history'].append(5.0)

# 性能显示文本对象
performance_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 仅在指定帧打印一次分步耗时明细
step_timing_printed = False

# 绘制事件钩子与待完成的绘制耗时测量
draw_hook_installed = False
pending_draw_measurement = None

def compute_arf_gpu_optimized(positions, time, domain_size, compute_sound_field, 
                              particle_diameters, is_standing_wave, batch_size=BATCH_SIZE):
    """高度优化的GPU ARF计算"""
    if not USE_GPU:
        # 如果不用GPU，回退到原始方法
        return pinn_arf.compute_arf_for_particles(
            positions, time, domain_size, compute_sound_field, 
            particle_diameters, is_standing_wave
        )
    
    N = positions.shape[0]
    
    # 预计算声场（只在GPU上计算一次）
    X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=time)
    P_tensor = torch.tensor(P, dtype=torch.float32, device=device)
    
    # 计算压力梯度（完全在GPU上）
    dx = domain_size[0] / (P.shape[1] - 1)
    dy = domain_size[1] / (P.shape[0] - 1)
    P_squared = P_tensor ** 2
    
    # 使用torch.gradient计算梯度
    grad_y, grad_x = torch.gradient(P_squared, spacing=(dy, dx))
    
    # 向量化插值计算（避免循环）
    x_vals = positions[:, 0]
    y_vals = positions[:, 1]
    
    # 计算插值索引
    x_indices = torch.clamp((x_vals / domain_size[0] * (P.shape[1] - 1)).long(), 0, P.shape[1] - 1)
    y_indices = torch.clamp((y_vals / domain_size[1] * (P.shape[0] - 1)).long(), 0, P.shape[0] - 1)
    
    # 批量获取梯度值（向量化操作）
    batch_grad_x = grad_x[y_indices, x_indices]
    batch_grad_y = grad_y[y_indices, x_indices]
    
    # 向量化ARF计算
    arf_strength = 1e1
    
    # 归一化梯度方向（向量化）
    grad_norm = torch.sqrt(batch_grad_x**2 + batch_grad_y**2)
    grad_norm = torch.clamp(grad_norm, min=1e-12)
    
    force_direction_x = -batch_grad_x / grad_norm
    force_direction_y = -batch_grad_y / grad_norm
    
    # 计算力的大小（基于粒子直径）
    force_magnitude = arf_strength * (particle_diameters ** 2)
    
    # 计算最终的ARF力（完全向量化）
    arf_forces = torch.stack([
        force_magnitude * force_direction_x,
        force_magnitude * force_direction_y
    ], dim=1)
    
    return arf_forces

def apply_stokes_drag_gpu(positions, velocities, radii, mass, dt, viscosity=1e-3):
    """高度优化的GPU斯托克斯阻力计算"""
    if not USE_GPU:
        # CPU版本
        return apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity)
    
    # GPU版本 - 完全向量化
    viscosity_tensor = torch.tensor(viscosity, dtype=torch.float32, device=device)
    drag_coeff = 6 * torch.pi * viscosity_tensor * radii
    drag_forces = -drag_coeff[:, None] * velocities
    velocity_changes = drag_forces * dt / mass[:, None]
    velocities += velocity_changes
    
    # 更新位置
    positions += velocities * dt
    
    return positions, velocities

def optimize_gpu_memory():
    """优化GPU内存使用"""
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def update_performance_display(frame):
    """更新性能显示"""
    if performance_stats['total_frames'] > 0:
        # 计算FPS
        current_time = time.time()
        frame_time = current_time - performance_stats['last_frame_time']
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # 限制FPS历史记录长度
        performance_stats['fps_history'].append(fps)
        if len(performance_stats['fps_history']) > 10:
            performance_stats['fps_history'].pop(0)
        
        # 获取系统指标（每帧都获取，但CPU只在特定间隔获取）
        if frame % 5 == 0:  # 每5帧获取一次CPU和内存，减少开销
            cpu_percent, memory_mb, memory_percent, total_memory_gb = get_system_metrics()
            performance_stats['cpu_history'].append(cpu_percent)
            performance_stats['memory_history'].append(memory_mb)
            performance_stats['memory_percent_history'].append(memory_percent)
            if len(performance_stats['cpu_history']) > 10:
                performance_stats['cpu_history'].pop(0)
                performance_stats['memory_history'].pop(0)
                performance_stats['memory_percent_history'].pop(0)
        else:
            # 使用上次的值
            cpu_percent = performance_stats['cpu_history'][-1] if performance_stats['cpu_history'] else 0.0
            memory_mb = performance_stats['memory_history'][-1] if performance_stats['memory_history'] else 0.0
            memory_percent = performance_stats['memory_percent_history'][-1] if performance_stats['memory_percent_history'] else 0.0
        
        # 计算平均值
        avg_fps = np.mean(performance_stats['fps_history'])
        avg_cpu = np.mean(performance_stats['cpu_history']) if performance_stats['cpu_history'] else 0.0
        avg_memory = np.mean(performance_stats['memory_history']) if performance_stats['memory_history'] else 0.0
        avg_memory_percent = np.mean(performance_stats['memory_percent_history']) if performance_stats['memory_percent_history'] else 0.0
        
        # 更新显示文本
        performance_text.set_text(
            f'Performance Metrics:\n'
            f'FPS: {fps:.1f} (Avg: {avg_fps:.1f})\n'
            f'CPU: {cpu_percent:.1f}% (Avg: {avg_cpu:.1f}%)\n'
            f'Memory: {memory_mb:.1f} MB ({memory_percent:.1f}%) (Avg: {avg_memory:.1f} MB, {avg_memory_percent:.1f}%), Total RAM: {performance_stats["total_memory_gb"]:.1f} GB\n'
            f'Particles: {N_particles}\n'
            f'Frames: {performance_stats["total_frames"]}'
        )
        
        performance_stats['last_frame_time'] = current_time

# 更新声场（首次计算在外部，随后每帧传入以避免重复计算）
X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=0.0)

def update(frame):
    global positions, velocities, radii, mass, performance_stats, P, step_timing_printed, draw_hook_installed, pending_draw_measurement
    
    import time
    start_time = time.perf_counter()
    t0 = start_time
    arf_time = 0.0
    
    dt = 1e-3  # 与DEM_main.py保持一致
    t = frame * 5e-7  # 当前时间
    
    # 每帧更新声场一次（CPU操作，但只做一次）
    Xp, Yp, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=t)
    sound_img.set_data(P)
    t1 = time.perf_counter()
    
    if USE_GPU:
        # GPU加速版本 - 完全在GPU上计算
        # 仅对ARF计算本身计时（调用前后同步）
        torch.cuda.synchronize()
        arf_start = time.perf_counter()
        
        # 使用优化的GPU ARF计算
        arf_forces = compute_arf_gpu_optimized(
            positions, t, domain_size, compute_sound_field, 
            particle_diameters, IS_STANDING_WAVE, batch_size=BATCH_SIZE
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        arf_time = t2 - t1
        
        # 应用ARF力到粒子速度（完全在GPU上）
        velocity_changes = arf_forces * dt / mass[:, None]
        velocities += velocity_changes
        
        # 应用斯托克斯阻力（GPU版本）
        positions, velocities = apply_stokes_drag_gpu(positions, velocities, radii, mass, dt, viscosity=1e-3)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        # 边界处理（周期性边界）
        positions[:, 0] = torch.fmod(positions[:, 0], domain_size[0])
        positions[:, 1] = torch.fmod(positions[:, 1], domain_size[1])
        
        # 同步GPU计算
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        
        # 更新粒子位置显示（只在需要时转换回CPU）
        if frame % 2 == 0:  # 每2帧更新一次显示，减少CPU-GPU传输
            scatter.set_offsets(positions.detach().cpu().numpy() * 1000)
        t5 = time.perf_counter()
        
        # 调试信息
        if frame % 20 == 0:  # 减少调试输出频率
            max_force = torch.max(torch.norm(arf_forces, dim=1)).item()
            max_vel_change = torch.max(torch.norm(velocity_changes, dim=1)).item()
            max_vel = torch.max(torch.norm(velocities, dim=1)).item()
            print(f"Frame {frame} (GPU): Max ARF = {max_force:.2e} N, "
                  f"Max vel change = {max_vel_change:.2e} m/s, Max vel = {max_vel:.2e} m/s")
        
        # 定期清理GPU内存
        if frame % 50 == 0:
            optimize_gpu_memory()
            
    else:
        # CPU版本
        arf_start = time.perf_counter()
        arf_forces = pinn_arf.compute_arf_for_particles(
            positions, t, domain_size, compute_sound_field, 
            particle_diameters, IS_STANDING_WAVE, precomputed_field=(Xp, Yp, P)
        )
        t2 = time.perf_counter()
        arf_time = t2 - t1
        
        velocity_changes = arf_forces * dt / mass[:, None]
        velocities += velocity_changes
        
        positions, velocities = apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1e-3)
        t3 = time.perf_counter()
        
        positions += velocities * dt
        positions[:, 0] = np.mod(positions[:, 0], domain_size[0])
        positions[:, 1] = np.mod(positions[:, 1], domain_size[1])
        
        scatter.set_offsets(positions * 1000)
        t5 = time.perf_counter()
        
        if frame % 20 == 0:
            max_force = np.max(np.linalg.norm(arf_forces, axis=1))
            max_vel_change = np.max(np.linalg.norm(velocity_changes, axis=1))
            max_vel = np.max(np.linalg.norm(velocities, axis=1))
            print(f"Frame {frame} (CPU): Max ARF = {max_force:.2e} N, "
                  f"Max vel change = {max_vel_change:.2e} m/s, Max vel = {max_vel:.2e} m/s")
    
    performance_stats['arf_calls'] += 1
    
    # 更新性能显示
    update_performance_display(frame)
    t6 = time.perf_counter()
    
    # 性能统计
    frame_time = time.perf_counter() - start_time
    performance_stats['total_frames'] += 1
    performance_stats['total_time'] += frame_time
    
    # 每50帧打印性能信息
    # 安装一次绘制完成的计时钩子（捕捉blit/绘制开销）
    if not draw_hook_installed:
        def _on_draw(event):
            global pending_draw_measurement
            if pending_draw_measurement is not None:
                pending_draw_measurement['draw_end'] = time.perf_counter()
        fig.canvas.mpl_connect('draw_event', _on_draw)
        draw_hook_installed = True

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
        t_boundary = ms(t4_l - t3_l)
        t_scatter = ms(t5_l - t4_l)
        t_perf = ms(t6_l - t5_l)
        draw_ms = ms(pending_draw_measurement['draw_end'] - pending_draw_measurement['draw_start'])
        total_ms = t_sound + t_arf + t_drag + t_boundary + t_scatter + t_perf + draw_ms

        print("\nStep timing breakdown (frame 5):")
        print(f"  Particles:          {N_particles}")
        print(f"  Sound field update: {t_sound:.2f} ms")
        print(f"  ARF computation:    {t_arf:.2f} ms")
        print(f"  Stokes drag:        {t_drag:.2f} ms")
        print(f"  Boundary wrap:      {t_boundary:.2f} ms")
        print(f"  Scatter update:     {t_scatter:.2f} ms")
        print(f"  Perf display:       {t_perf:.2f} ms")
        print(f"  Render (draw):      {draw_ms:.2f} ms")
        print(f"  Total (incl. draw): {total_ms:.2f} ms")
        step_timing_printed = True

    if frame % 50 == 0:
        avg_frame_time = performance_stats['total_time'] / performance_stats['total_frames']
        gpu_status = "GPU" if USE_GPU else "CPU"
        print(f"Frame {frame} ({gpu_status}): Avg frame time: {avg_frame_time*1000:.2f}ms, "
              f"ARF time: {arf_time*1000:.2f}ms, FPS: {1/avg_frame_time:.1f}")
    
    return sound_img, scatter, performance_text

# 性能对比测试
if PERFORMANCE_TEST:
    print("\n" + "="*60)
    print("性能对比测试模式")
    print("="*60)
    
    # 测试CPU性能
    print("测试CPU性能...")
    USE_GPU = False
    device = torch.device('cpu')
    positions_cpu = positions.cpu().numpy() if hasattr(positions, 'cpu') else positions
    velocities_cpu = velocities.cpu().numpy() if hasattr(velocities, 'cpu') else velocities
    radii_cpu = radii.cpu().numpy() if hasattr(radii, 'cpu') else radii
    mass_cpu = mass.cpu().numpy() if hasattr(mass, 'cpu') else mass
    
    cpu_start = time.time()
    for i in range(100):
        # 模拟一帧计算
        arf_forces = pinn_arf.compute_arf_for_particles(
            positions_cpu, 0.0, domain_size, compute_sound_field, 
            particle_diameters, IS_STANDING_WAVE
        )
        velocity_changes = arf_forces * 1e-3 / mass_cpu[:, None]
        velocities_cpu += velocity_changes
        positions_cpu += velocities_cpu * 1e-3
    cpu_time = time.time() - cpu_start
    
    # 测试GPU性能
    print("测试GPU性能...")
    USE_GPU = True
    device = torch.device('cuda')
    positions_gpu = torch.tensor(positions_cpu, dtype=torch.float32, device=device)
    velocities_gpu = torch.tensor(velocities_cpu, dtype=torch.float32, device=device)
    radii_gpu = torch.tensor(radii_cpu, dtype=torch.float32, device=device)
    mass_gpu = torch.tensor(mass_cpu, dtype=torch.float32, device=device)
    particle_diameters_gpu = torch.tensor(particle_diameters, dtype=torch.float32, device=device)
    
    gpu_start = time.time()
    for i in range(100):
        # 模拟一帧计算
        arf_forces = compute_arf_gpu_optimized(
            positions_gpu, 0.0, domain_size, compute_sound_field, 
            particle_diameters_gpu, IS_STANDING_WAVE, batch_size=BATCH_SIZE
        )
        velocity_changes = arf_forces * 1e-3 / mass_gpu[:, None]
        velocities_gpu += velocity_changes
        positions_gpu += velocities_gpu * 1e-3
    torch.cuda.synchronize()
    gpu_time = time.time() - gpu_start
    
    speedup = cpu_time / gpu_time
    print(f"\n性能对比结果:")
    print(f"CPU时间: {cpu_time*1000:.2f}ms (100帧)")
    print(f"GPU时间: {gpu_time*1000:.2f}ms (100帧)")
    print(f"加速比: {speedup:.2f}x")
    print(f"GPU性能提升: {(speedup-1)*100:.1f}%")
    print("="*60)
    
    # 恢复GPU模式
    positions = positions_gpu
    velocities = velocities_gpu
    radii = radii_gpu
    mass = mass_gpu
    particle_diameters = particle_diameters_gpu

# 创建动画
print(f"Starting PINN DEM simulation with {N_particles} particles...")
print(f"GPU Acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
if USE_GPU:
    print(f"Batch Size: {BATCH_SIZE}")
print("Press Ctrl+C to stop the simulation")

try:
    ani = animation.FuncAnimation(fig, update, frames=300, interval=33, blit=False)  # 与DEM_main.py保持一致
    plt.show()
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
finally:
    # 打印最终性能统计
    if performance_stats['total_frames'] > 0:
        avg_frame_time = performance_stats['total_time'] / performance_stats['total_frames']
        print(f"\nPerformance Summary:")
        print(f"Total frames: {performance_stats['total_frames']}")
        print(f"Average frame time: {avg_frame_time*1000:.2f}ms")
        print(f"FPS: {1/avg_frame_time:.1f}")
        print(f"ARF calculations: {performance_stats['arf_calls']}")
        
        # 打印性能指标总结
        if performance_stats['fps_history']:
            print(f"Average FPS: {np.mean(performance_stats['fps_history']):.1f}")
        if performance_stats['cpu_history']:
            print(f"Average CPU usage: {np.mean(performance_stats['cpu_history']):.1f}%")
        if performance_stats['memory_history']:
            print(f"Average memory usage: {np.mean(performance_stats['memory_history']):.1f} MB")
        if performance_stats['memory_percent_history']:
            print(f"Average memory percentage: {np.mean(performance_stats['memory_percent_history']):.1f}%")
        print(f"Total system RAM: {performance_stats['total_memory_gb']:.1f} GB")
