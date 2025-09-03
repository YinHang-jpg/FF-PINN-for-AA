# particle_initialization.py
import numpy as np

def initialize_particles(N=0, domain_size=(0.01, 0.01), diameter=2e-6, density=2000, frequency=10000, sound_speed=340):
    """
    初始化粒子群，在边界左右两边各扩展1/4个波长来放置粒子
    
    :param N: 粒子数量
    :param domain_size: 模拟域尺寸 (Lx, Ly)，单位：米
    :param diameter: 每个粒子的直径，单位：米
    :param density: 每个粒子的密度，单位：kg/m^3
    :param frequency: 声波频率，单位：Hz
    :param sound_speed: 声速，单位：m/s
    :return: 位置 (N,2)、速度 (N,2)、半径数组 (N,), 质量数组 (N,)
    """
    Lx, Ly = domain_size
    
    # 计算波长和1/4波长
    wavelength = sound_speed / frequency
    quarter_wavelength = wavelength / 4
    
    # 计算扩展后的总长度
    extended_length = Lx + 2 * quarter_wavelength
    
    # 在扩展长度上均匀分布粒子
    # 使用endpoint=True确保粒子分布在整个扩展域内，包括边界
    x_positions = np.linspace(-quarter_wavelength, Lx + quarter_wavelength, N, endpoint=True)
    y_position = Ly / 2  # y坐标固定在域中心
    
    positions = np.column_stack((x_positions, np.full(N, y_position)))

    velocities = np.zeros_like(positions)
    radii = np.ones(N) * (diameter / 2)
    # 计算质量（三维近似球体：m = ρ * 4/3 * π * r^3）
    mass = density * 4/3 * np.pi * radii**3
    
    return positions, velocities, radii, mass