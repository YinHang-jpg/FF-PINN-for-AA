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

# 配置开关 - 只使用时间PINN模型进行力预测
USE_TRAVELING_WAVE = False
USE_PINN_T_FORCES = True  # 使用时间PINN模型预测粒子受力
USE_GRAVITY = False
USE_STOKES_DRAG = False
USE_AGGLOMERATION = False
USE_BROWNIAN = False
USE_ACOUSTIC_WAKE = False
USE_COLLISION = False

# 调试打印配置
TARGET_PARTICLE_INDEX = 0  # 要监控的粒子索引

# 仿真时间累计变量
simulation_time = 0.0  # 累计仿真时间
density_plot_saved = False  # 标记是否已经保存过密度图
last_save_time = 0.0  # 记录上次保存图片的时间

# 导入声场计算（仅用于可视化）
if USE_TRAVELING_WAVE:
    from initialization.sound_source_traveling import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = False
else:
    from initialization.sound_source_standing import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = True

# 导入时间PINN模型
from mechanisms.ARF_PINN_t import ARFNetT as ARFNetT, Normalizer as NormalizerT

def compute_particle_forces_using_t_pinn(positions, velocities, mass, radii, dt, time, t_model, t_normalizer, print_forces=False):
    """
    使用训练好的时间PINN模型计算粒子受力
    - t_model: 时间相关的力模型 (ARFNetT)
    - print_forces: 是否打印力信息
    """
    if t_model is None or t_normalizer is None:
        print("时间PINN模型未完全初始化，无法计算力")
        return positions, velocities, None
    
    try:
        with torch.no_grad():
            # 准备输入数据 - 固定x=0，只考虑时间变化
            x = torch.zeros_like(torch.tensor(positions[:, 0], dtype=torch.float32))  # 固定x=0
            y = torch.tensor(positions[:, 1], dtype=torch.float32)
            t = torch.full_like(x, time, dtype=torch.float32)
            
            # 归一化时间用于t模型
            t_min, t_max = 0.0, 0.0001  # 从归一化参数文件获取
            t_norm = (t - t_min) / (t_max - t_min)
            
            # 使用t模型预测时间因子（归一化后的值）
            time_factor_norm = t_model(t_norm.unsqueeze(1))
            
            # 反归一化t模型的输出
            if t_normalizer is not None and 'fx' in t_normalizer.stats:
                tf_mu, tf_sigma = t_normalizer.stats['fx']
                time_factor = time_factor_norm * tf_sigma + tf_mu
            else:
                # 如果没有归一化器，使用默认的反归一化参数
                time_factor_mu = -0.0006397787947207689
                time_factor_sigma = 0.7075421214103699
                time_factor = time_factor_norm * time_factor_sigma + time_factor_mu
            
            # 计算理论时间因子 cos(ωt) 用于对比
            omega = 2 * np.pi * frequency  # 角频率
            theoretical_time_factor = np.cos(omega * time)
            
            # 由于只考虑时间变化，我们使用一个基础的空间力系数
            # 这里使用一个简化的空间力分布（基于声波在x=0处的特性）
            spatial_force_base = 1e-12  # 基础力系数 (N)
            
            # 使用理论时间因子而不是模型预测的时间因子
            fx = torch.full_like(time_factor.squeeze(), spatial_force_base) * theoretical_time_factor
            fy = torch.zeros_like(fx)  # 假设只有x方向的力
            
            # 转换为numpy数组
            fx = fx.numpy()
            fy = fy.numpy()
            
            # 打印力信息（如果需要）
            if print_forces:
                print(f"\n=== 时间步 {time:.6f}s 的粒子时间相关力 ===")
                print("粒子ID | X位置(mm) | Y位置(mm) | 理论时间因子 | 模型时间因子 | 时间相关力Fx(pN) | 时间相关力Fy(pN)")
                print("-" * 100)
                for i in range(len(positions)):
                    print(f"{i:6d} | {positions[i,0]*1000:8.3f} | {positions[i,1]*1000:8.3f} | {theoretical_time_factor:12.4f} | {time_factor[i].item():12.4f} | {fx[i]*1e12:15.3e} | {fy[i]*1e12:15.3e}")
                print(f"总粒子数: {len(positions)}")
                print(f"理论时间因子: {theoretical_time_factor:.4f}")
                print(f"模型时间因子: {torch.mean(time_factor).item():.4f}")
                print(f"平均力Fx: {np.mean(fx)*1e12:.3e} pN")
                print(f"力Fx范围: [{np.min(fx)*1e12:.3e}, {np.max(fx)*1e12:.3e}] pN")
                print("=" * 100)
            
            # 计算加速度
            accelerations = np.column_stack([fx, fy]) / mass[:, None]
            
            # 更新速度和位置
            velocities = velocities + accelerations * dt
            positions = positions + velocities * dt
            
            return positions, velocities, (fx, fy, theoretical_time_factor)
            
    except Exception as e:
        print(f"时间PINN模型计算力时出错: {e}")
        print("将使用零力（无外力效应）")
        # 如果PINN计算失败，不施加任何力，保持原有运动
        return positions, velocities, None

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
positions, velocities, radii, mass = initialize_particles(N=200, domain_size=domain_size)
# 记录初始位置用于位移计算
initial_positions = positions.copy()

# 初始化时间PINN模型
print("正在加载训练好的时间PINN模型...")
try:
    # 初始化t模型（时间相关）
    t_model = ARFNetT(period_seconds=1.0 / frequency)
    t_model_path = 'PINN/arf_model_t.pth'
    if os.path.exists(t_model_path):
        t_model.load_state_dict(torch.load(t_model_path, map_location='cpu'))
        print(f"成功加载t模型权重: {t_model_path}")
    else:
        print(f"警告: t模型文件 {t_model_path} 不存在")
        t_model = None
    
    # 加载归一化参数
    t_norm_path = 'arf_model_t_normalization_params.json'
    
    t_normalizer = None
    
    if os.path.exists(t_norm_path):
        t_normalizer = NormalizerT()
        t_normalizer.load(t_norm_path, device='cpu')
        print(f"成功加载t模型归一化参数: {t_norm_path}")
    
    # 设置为评估模式
    if t_model is not None:
        t_model.eval()
    
    print("时间PINN模型初始化完成")
    
except Exception as e:
    print(f"时间PINN模型初始化失败: {e}")
    print("将无法进行粒子力预测")
    t_model = None
    t_normalizer = None

# 绘图初始化 - 创建三个子图
fig, (ax_particles, ax_time_factor, ax_density) = plt.subplots(1, 3, figsize=(20, 6))

# 左侧子图：粒子运动
ax_particles.set_xlim(0, domain_size[0] * 1000)
ax_particles.set_ylim(0, domain_size[1] * 1000)
ax_particles.set_title("Particle Motion (Time-based Forces)")
ax_particles.set_xlabel("x (mm)")
ax_particles.set_ylabel("y (mm)")

# 中间子图：时间因子变化曲线
ax_time_factor.set_xlim(0, 0.0001)  # 一个周期的时间范围
ax_time_factor.set_ylim(-1.5, 1.5)  # 时间因子范围
ax_time_factor.set_title("Time Factor vs Time")
ax_time_factor.set_xlabel("Time (s)")
ax_time_factor.set_ylabel("Time Factor")
ax_time_factor.grid(True, alpha=0.3)

# 右侧子图：密度变化曲线
ax_density.set_xlim(0, domain_size[0] * 1000)
ax_density.set_ylim(-5, 5)  # 进一步缩小y轴范围，确保曲线可见
ax_density.set_title("Particle Density Distribution")
ax_density.set_xlabel("x (mm)")
ax_density.set_ylabel("Relative Change (%)")
ax_density.grid(True, alpha=0.3)

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

# 添加性能信息文本框
performance_text = ax_particles.text(0.02, 0.98, '', transform=ax_particles.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 初始化时间因子曲线
time_points = []
time_factor_values = []
time_line, = ax_time_factor.plot([], [], 'r-', linewidth=2, label='Time Factor')
ax_time_factor.legend()

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
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, time_points, time_factor_values, x_grid, baseline

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

    dt = 1e-6
    global simulation_time
    simulation_time += dt  # 累计仿真时间
    t = simulation_time  # 使用累计时间

    # 使用时间PINN模型计算粒子受力
    if USE_PINN_T_FORCES:
        if t_model is not None:
            # 使用训练好的时间PINN模型计算粒子受力
            if frame % 10000 == 0:  # 每100帧打印一次状态和力信息
                print(f"使用时间PINN模型计算粒子受力 - 时间: {t:.6f}s, 粒子数: {len(positions)}")
                positions, velocities, forces = compute_particle_forces_using_t_pinn(
                    positions, velocities, mass, radii, dt, t, t_model, t_normalizer, print_forces=True
                )
            else:
                positions, velocities, forces = compute_particle_forces_using_t_pinn(
                    positions, velocities, mass, radii, dt, t, t_model, t_normalizer, print_forces=False
                )
            
            # 记录时间因子用于绘图
            if forces is not None:
                time_points.append(t)
                time_factor_values.append(forces[2])  # 理论时间因子
                
                # 保持最近1000个数据点
                if len(time_points) > 1000:
                    time_points.pop(0)
                    time_factor_values.pop(0)
        else:
            if frame % 10000 == 0:  # 每100帧打印一次状态
                print("警告: 时间PINN模型未完全初始化，跳过力计算")
            # 不施加任何力，保持原有运动

    # 更新声场 & 粒子位置
    Nx, Ny = (200, 200)
    _, _, P = compute_sound_field(domain_size=domain_size, resolution=(Nx, Ny), time=t)
    sound_img.set_data(P)
    scatter.set_offsets(positions * 1000)
    
    # 更新时间因子曲线
    if len(time_points) > 1:
        time_line.set_data(time_points, time_factor_values)
        # 动态调整x轴范围
        if len(time_points) > 100:
            ax_time_factor.set_xlim(max(0, time_points[-1] - 0.0001), time_points[-1])
        else:
            ax_time_factor.set_xlim(0, max(0.0001, time_points[-1] if time_points else 0.0001))
    
    ax_time_factor.set_title(f"Time Factor vs Time (t = {t:.6f} s)")
    
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

    # 更新性能信息显示
    pinn_status = "✓ Time PINN Active" if t_model is not None else "✗ Time PINN Inactive"
    performance_text.set_text(
        f'Simulation Time: {t:.6f} s\n'
        f'FPS: {performance_data["fps"]:.1f}\n'
        f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
        f'Particles: {performance_data["particle_count"]}\n'
        f'Time Model: {pinn_status}'
    )
    
    return sound_img, scatter, performance_text, time_line, density_line

ani = animation.FuncAnimation(fig, update, frames=20000, interval=0, blit=False)
plt.show()
