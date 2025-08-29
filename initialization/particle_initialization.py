# particle_initialization.py
import numpy as np

def initialize_particles(N=100, domain_size=(0.01, 0.01), diameter=2e-6, density=2000):
    """
    初始化粒子群
    :param N: 粒子数量
    :param domain_size: 模拟域尺寸 (Lx, Ly)，单位：米
    :param diameter: 每个粒子的直径，单位：米
    :param density: 每个粒子的密度，单位：kg/m^3
    :return: 位置 (N,2)、速度 (N,2)、半径数组 (N,), 质量数组 (N,)
    """
    Lx, Ly = domain_size
    # 将粒子均匀分布在一横行上，y坐标固定在域的中心
    x_positions = np.linspace(0, Lx, N, endpoint=False)  # 均匀分布x坐标
    y_position = Ly / 2  # y坐标固定在域中心
    
    positions = np.column_stack((x_positions, np.full(N, y_position)))

    velocities = np.zeros_like(positions)
    radii = np.ones(N) * (diameter / 2)
    # 计算质量（三维近似球体：m = ρ * 4/3 * π * r^3）
    mass = density * 4/3 * np.pi * radii**3
    return positions, velocities, radii, mass