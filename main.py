import matplotlib.pyplot as plt
import matplotlib.animation as animation
from initialization.particle_initialization import initialize_particles
import numpy as np
from scipy.spatial import cKDTree
from mechanisms.wake import WAKE_STRENGTH_MULTIPLIER
import psutil
import time
import threading
import os

# 配置开关
USE_TRAVELING_WAVE = False
USE_TEST_PARTICLE = False
USE_ARF = True
USE_GRAVITY = False
USE_STOKES_DRAG = True
USE_AGGLOMERATION = False
USE_BROWNIAN = False
USE_ACOUSTIC_WAKE = True  # 开启尾流场影响
USE_COLLISION = False  # 开启碰撞处理

# 仿真时间累计变量
simulation_time = 0.0  # 累计仿真时间
density_plot_saved = False  # 标记是否已经保存过密度图
last_save_time = 0.0  # 记录上次保存图片的时间



if USE_TRAVELING_WAVE:
    from initialization.sound_source_traveling import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = False
else:
    from initialization.sound_source_standing import compute_sound_field, frequency, amplitude
    IS_STANDING_WAVE = True

from mechanisms.ARF import compute_pressure_gradient_and_apply_arf
from mechanisms.gravity import apply_gravity
from mechanisms.Stokes_drag import apply_stokes_drag
from mechanisms.agglomeration import apply_agglomeration
from mechanisms.Brownian import apply_brownian_motion
from mechanisms.wake import acoustic_wake_velocity
from mechanisms.collision import apply_collision_physics, calculate_merged_particle_properties
from initialization.test_particle import initialize_test_particle, update_test_particle_position, get_test_particle_info

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
positions, velocities, radii, mass = initialize_particles(N=1000, domain_size=domain_size)

# 测试粒子
if USE_TEST_PARTICLE:
    test_positions, test_velocities, test_radii, test_mass = initialize_test_particle(
        domain_size=domain_size,
        speed=0.05,
        diameter=2e-6,
        density=2000
    )
    test_particle_info = get_test_particle_info()

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

if USE_TEST_PARTICLE:
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Regular Particles'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=test_particle_info['color'], markersize=8, label=test_particle_info['name']),
        Line2D([0], [0], marker='', color='green', linewidth=2, label='Wake Velocity Field')
    ]
    ax_particles.legend(handles=legend_elements, loc='upper right')

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

# 测试粒子 + 尾流箭头
if USE_TEST_PARTICLE:
    test_scatter = ax_particles.scatter(test_positions[:, 0] * 1000,
                              test_positions[:, 1] * 1000,
                              s=(test_radii * 1e6 * 4)**2,
                              c=test_particle_info['color'],
                              alpha=0.8,
                              marker=test_particle_info['marker'])

    wake_resolution = 20  # 增加分辨率
    wake_x = np.linspace(-0.003, 0.003, wake_resolution)  # 调整网格范围
    wake_y = np.linspace(-0.003, 0.003, wake_resolution)
    wake_X, wake_Y = np.meshgrid(wake_x, wake_y)

    wake_quiver = ax_particles.quiver(wake_X * 1000, wake_Y * 1000,
                            np.zeros_like(wake_X), np.zeros_like(wake_Y),
                            np.zeros_like(wake_X),
                            cmap='viridis', scale=0.1, scale_units='width',  # 调整scale参数
                            angles='xy', width=0.004, alpha=0.7)  # 减小width，让箭头更窄

    wake_cbar = plt.colorbar(wake_quiver, ax=ax_particles, shrink=0.8)
    wake_cbar.set_label('Wake Velocity Magnitude', fontsize=10)

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
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved
    if USE_TEST_PARTICLE:
        global test_positions, test_velocities, test_radii, test_mass, wake_X, wake_Y, wake_quiver

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

    dt = 1e-7
    global simulation_time
    simulation_time += dt  # 累计仿真时间
    t = simulation_time  # 使用累计时间

    # 力学更新
    if USE_STOKES_DRAG:
        # 计算测试粒子对周围粒子的斯托克斯力影响
        if USE_TEST_PARTICLE and USE_ACOUSTIC_WAKE:
            # 为每个普通粒子计算尾流场的影响
            for i in range(len(positions)):
                # 计算测试粒子到普通粒子的距离
                rel_pos = positions[i] - test_positions[0]
                distance = np.linalg.norm(rel_pos)
                
                # 如果距离足够近，考虑尾流场的影响
                if distance < 0.005:  # 5mm范围内
                    # 计算尾流速度场
                    wake_vel = acoustic_wake_velocity(
                        source_pos=test_positions[0],
                        target_pos=positions[i],
                        source_vel=test_velocities[0],
                        Re=10,
                        return_cartesian=True
                    )
                    
                    # 将尾流速度作为额外的斯托克斯力
                    if np.linalg.norm(wake_vel) > 1e-6:
                        # 计算斯托克斯阻力系数
                        particle_radius = radii[i]
                        fluid_viscosity = 1.79e-5
                        stokes_coeff = 6 * np.pi * fluid_viscosity * particle_radius
                        
                        # 应用尾流场产生的力
                        wake_force = stokes_coeff * np.array(wake_vel)
                        velocities[i] += wake_force * dt / mass[i]
        
        # 原有的斯托克斯阻力
        positions, velocities = apply_stokes_drag(positions, velocities, radii, mass, dt)

    # 应用声辐射力 (ARF)
    if USE_ARF:
        # 将ARF计算与方向判定完全交由 ARF.py 处理
        positions, velocities = compute_pressure_gradient_and_apply_arf(
            positions, velocities, mass, dt, domain_size, (200, 200), t, compute_sound_field,
            arf_strength=1e-10, is_standing_wave=IS_STANDING_WAVE
        )

    # 应用重力
    if USE_GRAVITY:
        positions, velocities = apply_gravity(positions, velocities, radii, mass, dt)

    # 应用碰撞物理
    if USE_COLLISION:
        positions, velocities, radii, mass = apply_collision_physics(
            positions, velocities, radii, mass, dt
        )



    if USE_TEST_PARTICLE:
        test_positions = update_test_particle_position(test_positions, test_velocities, domain_size, dt)

        # 不再使用固定箭头长度，而是根据速度大小动态调整
        # arrow_len = 0.005  # 注释掉固定长度

        # 创建全局网格（跟随粒子中心）
        # 改用极坐标网格，确保与test_wake.py中的方向完全一致
        wake_r = np.linspace(0.001, 0.003, wake_resolution)  # 径向距离，避免r=0
        wake_theta = np.linspace(-np.pi, np.pi, wake_resolution)  # 角度范围
        wake_R, wake_Theta = np.meshgrid(wake_r, wake_theta)
        
        # 转换为笛卡尔坐标（相对于粒子中心）
        wake_X = wake_R * np.cos(wake_Theta)
        wake_Y = wake_R * np.sin(wake_Theta)
        
        # 全局坐标（跟随粒子）
        abs_wake_X = wake_X + test_positions[0, 0]
        abs_wake_Y = wake_Y + test_positions[0, 1]

        wake_Vx = np.zeros_like(wake_X)
        wake_Vy = np.zeros_like(wake_Y)
        wake_speed = np.zeros_like(wake_X)  # 只是颜色映射

        for i in range(wake_X.shape[0]):
            for j in range(wake_X.shape[1]):
                grid_pos = np.array([abs_wake_X[i, j], abs_wake_Y[i, j]])
                wake_vel = acoustic_wake_velocity(
                    source_pos=test_positions[0],
                    target_pos=grid_pos,
                    source_vel=test_velocities[0],
                    Re=10,
                    return_cartesian=True
                )

                norm = np.linalg.norm(wake_vel)
                if norm > 1e-6:  # 只有当速度足够大时才显示箭头
                    dir_vec = wake_vel / norm
                    # 根据速度大小动态调整箭头长度
                    arrow_len = min(0.003, max(0.0005, norm * 0.1))  # 限制在0.5mm到3mm之间
                    wake_Vx[i, j] = dir_vec[0] * arrow_len
                    wake_Vy[i, j] = dir_vec[1] * arrow_len
                    wake_speed[i, j] = norm  # 颜色还是用真实速度
                else:
                    # 如果尾流速度太小，使用一个小的默认箭头
                    vel_unit = test_velocities[0] / (np.linalg.norm(test_velocities[0]) + 1e-12)
                    arrow_len = 0.0005  # 默认箭头长度0.5mm
                    wake_Vx[i, j] = vel_unit[0] * arrow_len * 0.3
                    wake_Vy[i, j] = vel_unit[1] * arrow_len * 0.3
                    wake_speed[i, j] = 0.01

        # 确保至少有一些箭头显示
        if np.all(wake_Vx == 0) and np.all(wake_Vy == 0):
            # 如果所有箭头都是零，创建一些默认箭头
            for i in range(wake_X.shape[0]):
                for j in range(wake_X.shape[1]):
                    if i % 4 == 0 and j % 4 == 0:  # 每隔几个点显示一个箭头
                        wake_Vx[i, j] = test_velocities[0, 0] * arrow_len * 0.2
                        wake_Vy[i, j] = test_velocities[0, 1] * arrow_len * 0.2
                        wake_speed[i, j] = 0.01

        # 添加调试信息
        if frame % 100 == 0:  # 每100帧打印一次调试信息
            print(f"Frame {frame}: Max wake speed: {np.max(wake_speed):.6f}")
            print(f"Max wake Vx: {np.max(np.abs(wake_Vx)):.6f}, Max wake Vy: {np.max(np.abs(wake_Vy)):.6f}")
            print(f"Non-zero arrows: {np.sum(wake_Vx != 0) + np.sum(wake_Vy != 0)}")
            print(f"Arrow length: {arrow_len}")

        # 更新箭头显示（坐标单位转成 mm）
        wake_quiver.set_offsets(np.c_[abs_wake_X.ravel() * 1000, abs_wake_Y.ravel() * 1000])
        wake_quiver.set_UVC(wake_Vx.ravel(), wake_Vy.ravel(), wake_speed.ravel())

    # 更新声场 & 粒子位置
    _, _, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=t)
    sound_img.set_data(P)
    scatter.set_offsets(positions * 1000)
    
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
    
    # 每0.01s保存一次密度分布图，从0.01s到0.2s
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
        
        print(f"密度分布图已保存: {filename}")
        print(f"仿真时间: {t:.6f} s")
        print(f"最大相对变化: {y_max:.2f}%")
        print(f"最小相对变化: {y_min:.2f}%")
        print(f"已保存图片数量: {int(t * 100)}")
        print("-" * 50)
        
        last_save_time = t  # 更新上次保存时间

    if USE_TEST_PARTICLE:
        test_scatter.set_offsets(test_positions * 1000)
        
        # 更新性能信息显示
        performance_text.set_text(
            f'Simulation Time: {t:.6f} s\n'
            f'FPS: {performance_data["fps"]:.1f}\n'
            f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
            f'Particles: {performance_data["particle_count"]}'
        )
        
        return sound_img, scatter, test_scatter, wake_quiver, performance_text, density_line
    else:
        # 更新性能信息显示
        performance_text.set_text(
            f'Simulation Time: {t:.6f} s\n'
            f'FPS: {performance_data["fps"]:.1f}\n'
            f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
            f'Particles: {performance_data["particle_count"]}'
        )
        
        return sound_img, scatter, performance_text, density_line


ani = animation.FuncAnimation(fig, update, frames=20000, interval=33, blit=False)
plt.show()
