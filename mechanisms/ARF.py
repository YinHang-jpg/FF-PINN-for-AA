import numpy as np
from initialization.sound_source_standing import sound_pressure_level, spl_to_pressure, compute_sound_field

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

# 动态计算声压幅值：2倍声压（从声压级转换）
def get_amplitude():
    """
    计算声压幅值，等于2倍从声压级转换得到的声压值
    :return: 声压幅值 (Pa)
    """
    sound_pressure = 2 * spl_to_pressure(sound_pressure_level)
    A = sound_pressure/rho_0/c_0/2/np.pi/frequency
    return A

def get_particle_arf_force(particle_x, particle_y, particle_radius, time, domain_size=(0.034, 0.034)):
    """
    获取特定粒子的声辐射力
    
    :param particle_x: 粒子x坐标
    :param particle_y: 粒子y坐标
    :param particle_radius: 粒子半径
    :param time: 当前时间
    :param domain_size: 域大小，用于声场计算
    :return: ARF力向量 [Fx, Fy]
    """
    # 粒子直径
    d_p = 2 * particle_radius
    
    # 计算粒子前后两个点的位置（沿x方向，因为驻波是x方向的）
    # 根据公式：x-√2d_p/4 和 x+√2d_p/4
    offset = np.sqrt(2) * d_p / 4
    
    # 前点位置
    x_front = particle_x - offset
    # 后点位置  
    x_back = particle_x + offset
    
    # 使用sound_source_standing.py的声场计算函数获取声压
    # 计算整个声场
    _, _, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=time)
    
    # 通过插值获取粒子前后两点的声压值
    Lx, Ly = domain_size
    x_grid = np.linspace(0, Lx, 200)
    
    # 插值获取声压值（使用y=0的切片，因为驻波沿x方向）
    p_front = np.interp(x_front, x_grid, P[0, :])
    p_back = np.interp(x_back, x_grid, P[0, :])
    
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
    positions, radii, time
):
    """计算并返回每个粒子的声辐射力（不在此处更新速度与位置）。

    使用解析公式：F_p = π d_p^2 [p(x-√2 d_p/4, t) - p(x+√2 d_p/4, t)] / 4

    返回值：
    - arf_force: (N,2) 每个粒子的ARF力向量
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

        # 前点位置和后点位置
        x_front = x - offset
        x_back = x + offset

        # 使用解析公式计算声压
        # p(x,t) = 2πAρ₀γsin(2πx/λ)cos(2πft)/λ
        k = 2 * np.pi / wavelength  # 波数
        omega = 2 * np.pi * frequency  # 角频率
        amplitude = get_amplitude()  # 动态获取声压幅值

        p_front = 2 * 2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_front) * np.cos(omega * time) / wavelength
        p_back =  2 * 2 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_back ) * np.cos(omega * time) / wavelength

        # 计算压力差
        pressure_diff = p_front - p_back

        # 根据公式计算力：F_p = πd_p²[p_front - p_back]/4
        force_magnitude = np.pi * d_p**2 * pressure_diff / 4

        # 力的方向：沿x轴
        arf_force[i, 0] = force_magnitude
        arf_force[i, 1] = 0.0

    return arf_force
