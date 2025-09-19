import numpy as np

def bilinear_interpolate(grid, x, y, dx, dy):
    Nx = grid.shape[1]
    Ny = grid.shape[0]
    ix = np.clip(x / dx, 0, Nx - 2)
    iy = np.clip(y / dy, 0, Ny - 2)
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    x1 = np.clip(x0 + 1, 0, Nx - 1)
    y1 = np.clip(y0 + 1, 0, Ny - 1)
    wx = ix - x0
    wy = iy - y0
    # grid[y, x]，所以先行后列
    return (
        (1-wx)*(1-wy)*grid[y0, x0] +
        wx*(1-wy)*grid[y0, x1] +
        (1-wx)*wy*grid[y1, x0] +
        wx*wy*grid[y1, x1]
    )
p_0 = 101325    # Pa
# 物理参数（空气）
rho_0 = 1.225  # 空气密度 (kg/m³)
gamma = 1.4    # 空气绝热指数
c_0 = 340      # 声速 (m/s)
frequency = 10000  # Hz
amplitude = 7.64e-6    # m
def get_particle_arf_force(particle_x, particle_y, particle_radius, time):
    """
    获取特定粒子的声辐射力
    
    :param particle_x: 粒子x坐标
    :param particle_y: 粒子y坐标
    :param particle_radius: 粒子半径
    :param time: 当前时间
    :return: ARF力向量 [Fx, Fy]
    """
    # 计算波长和波数
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    
    # 粒子直径
    d_p = 2 * particle_radius
    
    # 计算粒子前后两个点的位置（沿x方向，因为驻波是x方向的）
    # 根据公式：x-√2d_p/4 和 x+√2d_p/4
    offset = np.sqrt(2) * d_p / 4
    
    # 前点位置
    x_front = particle_x - offset
    # 后点位置  
    x_back = particle_x + offset
    
    # 使用解析公式计算声压
    # p(x,t) = 2πAρ₀γsin(2πx/λ)cos(2πft)/λ
    p_front = 2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_front) * np.cos(omega * time) / wavelength
    p_back  = 2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_back ) * np.cos(omega * time) / wavelength
    
    # 计算压力差
    pressure_diff = p_front - p_back
    
    # 根据公式计算力：F_p = πd_p²[p_front - p_back]/4
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4
    
    # 力的方向：从高压指向低压（沿x轴方向）
    # 如果p_front > p_back，力向右（正x方向）
    # 如果p_front < p_back，力向左（负x方向）
    arf_force = np.array([force_magnitude, 0.0])  # 只有x方向分量
    
    return arf_force

def compute_pressure_gradient_and_apply_arf(
    positions, velocities, mass, radii, dt, domain_size, resolution, time, compute_sound_field, is_standing_wave=False
):
    """使用图片中的解析公式计算声辐射力：F_p = πd_p²[p(x-√2d_p/4,t) - p(x+√2d_p/4,t)]/4
    
    其中：
    - 声压：p(x,t) = 2πAρ₀γsin(2πx/λ)cos(2πft)/λ
    - A: 声压振幅 (Pa)
    - ρ₀: 介质密度 (kg/m³)
    - γ: 绝热指数 (比热比)
    - λ: 波长 (m)
    - f: 频率 (Hz)
    """
    
    # 计算波长
    wavelength = c_0 / frequency
    
    # 计算每个粒子的ARF
    arf_force = np.zeros_like(positions)
    for i, (x, y) in enumerate(positions):
        # 粒子直径
        d_p = 2 * radii[i]
        
        # 计算粒子前后两个点的位置（沿x方向，因为驻波是x方向的）
        # 根据公式：x-√2d_p/4 和 x+√2d_p/4
        offset = np.sqrt(2) * d_p / 4
        
        # 前点位置
        x_front = x - offset
        # 后点位置  
        x_back = x + offset
        
        
        # 使用解析公式计算声压
        # p(x,t) = 2πAρ₀γsin(2πx/λ)cos(2πft)/λ
        k = 2 * np.pi / wavelength  # 波数
        omega = 2 * np.pi * frequency  # 角频率
        
        p_front = 2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_front) * np.cos(omega * time) / wavelength
        p_back =  2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_back ) * np.cos(omega * time) / wavelength

        # 计算压力差
        pressure_diff = p_front - p_back

        # 根据公式计算力：F_p = πd_p²[p_front - p_back]/4
        force_magnitude = np.pi * d_p**2 * pressure_diff / 4
        
        # 力的方向：从高压指向低压（沿x轴方向）
        # 如果p_front > p_back，力向右（正x方向）
        # 如果p_front < p_back，力向左（负x方向）
        arf_force[i, 0] = force_magnitude  # x方向分量
        arf_force[i, 1] = 0.0              # y方向分量（驻波沿x方向，y方向无梯度）
        if time == 5e-7:
            print(x*1000, p_front, p_back, pressure_diff, force_magnitude, mass[i], force_magnitude/mass[i])
    # 加速度 = F / m
    accelerations = arf_force / mass[:, None]

    # 显式欧拉更新
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt

    return positions, velocities
