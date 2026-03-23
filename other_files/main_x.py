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

# 配置开关 - 只使用位置PINN模型进行力预测
USE_TRAVELING_WAVE = False
USE_PINN_X_FORCES = True  # 使用位置PINN模型预测粒子受力
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

# 导入位置PINN模型
from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as NormalizerX

def compute_particle_forces_using_x_pinn(positions, velocities, mass, radii, dt, time, x_model, x_normalizer, print_forces=False):
    """
    使用训练好的位置PINN模型计算粒子受力
    - x_model: 位置相关的力模型 (ARFNetX)
    - print_forces: 是否打印力信息
    """
    if x_model is None or x_normalizer is None:
        print("位置PINN模型未完全初始化，无法计算力")
        return positions, velocities, None
    
    try:
        with torch.no_grad():
            # 准备输入数据 - 使用实际x位置，固定t=0
            x = torch.tensor(positions[:, 0], dtype=torch.float32)
            y = torch.tensor(positions[:, 1], dtype=torch.float32)
            t = torch.zeros_like(x)  # 固定t=0
            
            # 将坐标中心化到通道中点，再按训练范围归一化
            x_centered = x - (domain_size[0] / 2.0)
            x_norm = (x_centered - X_MIN) / (X_MAX - X_MIN)
            x_norm = torch.clamp(x_norm, 0.0, 1.0)
            
            # 使用x模型预测位置相关的力（归一化后的值）
            fx_spatial_norm = x_model(x_norm.unsqueeze(1))
            
            # 反归一化x模型的输出
            if x_normalizer is not None and 'fx' in x_normalizer.stats:
                fx_mu, fx_sigma = x_normalizer.stats['fx']
                fx_spatial = fx_spatial_norm * fx_sigma + fx_mu
            else:
                # 如果没有归一化器，使用加载/默认的反归一化参数
                fx_spatial = fx_spatial_norm * FORCE_SIGMA + FORCE_MU
            
            # 由于固定t=0，时间因子为1（cos(0) = 1）
            time_factor = 1.0
            
            # 组合得到最终的力：F = F_spatial * F_temporal (t=0时F_temporal=1)
            fx = fx_spatial.squeeze() * time_factor
            fy = torch.zeros_like(fx)  # 假设只有x方向的力
            
            # 转换为numpy数组
            fx = fx.numpy()
            fy = fy.numpy()
            
            # 打印力信息（如果需要）
            if print_forces:
                print(f"\n=== 时间步 {time:.6f}s 的粒子位置相关力 ===")
                print("粒子ID | X位置(mm) | Y位置(mm) | 位置相关力Fx(pN) | 位置相关力Fy(pN)")
                print("-" * 80)
                for i in range(len(positions)):
                    print(f"{i:6d} | {positions[i,0]*1000:8.3f} | {positions[i,1]*1000:8.3f} | {fx[i]*1e12:15.3e} | {fy[i]*1e12:15.3e}")
                print(f"总粒子数: {len(positions)}")
                print(f"时间因子 (t=0): {time_factor:.4f}")
                print(f"平均力Fx: {np.mean(fx)*1e12:.3e} pN")
                print(f"力Fx范围: [{np.min(fx)*1e12:.3e}, {np.max(fx)*1e12:.3e}] pN")
                print("=" * 80)
            
            # 计算加速度
            accelerations = np.column_stack([fx, fy]) / mass[:, None]
            
            # 更新速度和位置
            velocities = velocities + accelerations * dt
            positions = positions + velocities * dt
            
            return positions, velocities, (fx, fy)
            
    except Exception as e:
        print(f"位置PINN模型计算力时出错: {e}")
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
positions, velocities, radii, mass = initialize_particles(N=50, domain_size=domain_size)
# 记录初始位置用于位移计算
initial_positions = positions.copy()

# 初始化位置PINN模型
print("正在加载训练好的位置PINN模型...")
try:
    # 初始化x模型（位置相关）
    x_model = ARFNetX(fourier_features=32)
    x_model_path = 'PINN/arf_model_x.pth'
    if os.path.exists(x_model_path):
        x_model.load_state_dict(torch.load(x_model_path, map_location='cpu'))
        print(f"成功加载x模型权重: {x_model_path}")
    else:
        print(f"警告: x模型文件 {x_model_path} 不存在")
        x_model = None
    
    # 加载归一化参数
    x_norm_path = 'PINN/arf_model_x_normalization_params.json'
    
    x_normalizer = None
    # 默认参数（若文件缺失时使用）
    X_MIN = -0.0255
    X_MAX = 0.0255
    FORCE_MU = 4.5863430241099845e-12
    FORCE_SIGMA = 1.4498783250382896e-11
    
    if os.path.exists(x_norm_path):
        x_normalizer = NormalizerX()
        x_normalizer.load(x_norm_path, device='cpu')
        try:
            with open(x_norm_path, 'r') as f:
                _params = json.load(f)
            if 'x_min' in _params and 'x_max' in _params:
                X_MIN = float(_params['x_min'])
                X_MAX = float(_params['x_max'])
            if 'force_mu' in _params:
                FORCE_MU = float(_params['force_mu'])
            if 'force_sigma' in _params:
                FORCE_SIGMA = max(float(_params['force_sigma']), 1e-30)
            print(f"成功加载x模型归一化参数: {x_norm_path}")
            print(f"x范围: [{X_MIN*1000:.1f}, {X_MAX*1000:.1f}] mm | 力均值/方差: {FORCE_MU:.3e}/{FORCE_SIGMA:.3e}")
        except Exception as _e:
            print(f"读取x归一化参数文件时出错: {_e}，使用默认参数")
    
    # 设置为评估模式
    if x_model is not None:
        x_model.eval()
    
    print("位置PINN模型初始化完成")
    
except Exception as e:
    print(f"位置PINN模型初始化失败: {e}")
    print("将无法进行粒子力预测")
    x_model = None
    x_normalizer = None

# 绘图初始化 - 创建三个子图
fig, (ax_particles, ax_force_profile, ax_density) = plt.subplots(1, 3, figsize=(20, 6))

# 左侧子图：粒子运动
ax_particles.set_xlim(0, domain_size[0] * 1000)
ax_particles.set_ylim(0, domain_size[1] * 1000)
ax_particles.set_title("Particle Motion (Position-based Forces)")
ax_particles.set_xlabel("x (mm)")
ax_particles.set_ylabel("y (mm)")

# 中间子图：位置相关力分布曲线
ax_force_profile.set_xlim(-domain_size[0] * 1000/2, domain_size[0] * 1000/2)  # x位置范围
ax_force_profile.set_ylim(-50, 50)  # 力值范围 (pN)
ax_force_profile.set_title("Force Profile vs Position (t=0)")
ax_force_profile.set_xlabel("Position x (mm)")
ax_force_profile.set_ylabel("Force Fx (pN)")
ax_force_profile.grid(True, alpha=0.3)

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

# 初始化力分布曲线
x_positions = np.linspace(-domain_size[0]/2, domain_size[0]/2, 200)  # 位置范围
force_profile = np.zeros_like(x_positions)  # 力分布
force_line, = ax_force_profile.plot(x_positions * 1000, force_profile, 'r-', linewidth=2, label='Force Profile')
ax_force_profile.legend()

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
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, x_positions, force_profile, x_grid, baseline

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

    # 使用位置PINN模型计算粒子受力
    if USE_PINN_X_FORCES:
        if x_model is not None:
            # 使用训练好的位置PINN模型计算粒子受力
            if frame % 10000 == 0:  # 每10000帧打印一次状态和力信息
                print(f"使用位置PINN模型计算粒子受力 - 时间: {t:.6f}s, 粒子数: {len(positions)}")
                positions, velocities, forces = compute_particle_forces_using_x_pinn(
                    positions, velocities, mass, radii, dt, t, x_model, x_normalizer, print_forces=True
                )
            else:
                positions, velocities, forces = compute_particle_forces_using_x_pinn(
                    positions, velocities, mass, radii, dt, t, x_model, x_normalizer, print_forces=False
                )
            
            # 计算力分布用于绘图
            if forces is not None and frame % 100 == 0:  # 每100帧更新一次力分布
                # x_positions 已以通道中心为0，直接用训练范围归一化
                x_norm_vals = (x_positions - X_MIN) / (X_MAX - X_MIN)
                x_norm_vals = np.clip(x_norm_vals, 0.0, 1.0)
                x_tensor = torch.tensor(x_norm_vals, dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    fx_spatial_norm = x_model(x_tensor)
                    force_profile = (fx_spatial_norm.squeeze() * FORCE_SIGMA + FORCE_MU).numpy() * 1e12  # 转换为pN
        else:
            if frame % 10000 == 0:  # 每10000帧打印一次状态
                print("警告: 位置PINN模型未完全初始化，跳过力计算")
            # 不施加任何力，保持原有运动

    # 更新声场 & 粒子位置
    Nx, Ny = (200, 200)
    _, _, P = compute_sound_field(domain_size=domain_size, resolution=(Nx, Ny), time=t)
    sound_img.set_data(P)
    scatter.set_offsets(positions * 1000)
    
    # 更新力分布曲线
    force_line.set_data(x_positions * 1000, force_profile)
    
    # 动态调整y轴范围
    if len(force_profile) > 0:
        y_min = np.min(force_profile)
        y_max = np.max(force_profile)
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax_force_profile.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax_force_profile.set_title(f"Force Profile vs Position (t = {t:.6f} s)")
    
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
    pinn_status = "✓ Position PINN Active" if x_model is not None else "✗ Position PINN Inactive"
    performance_text.set_text(
        f'Simulation Time: {t:.6f} s\n'
        f'FPS: {performance_data["fps"]:.1f}\n'
        f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
        f'Particles: {performance_data["particle_count"]}\n'
        f'Position Model: {pinn_status}'
    )
    
    return sound_img, scatter, performance_text, force_line, density_line

ani = animation.FuncAnimation(fig, update, frames=20000, interval=0, blit=False)
plt.show()
