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
    np.random.seed(0)
    positions = np.random.rand(N, 2) * [Lx, Ly]
    velocities = np.zeros_like(positions)
    radii = np.ones(N) * (diameter / 2)
    # 计算质量（二维近似为圆盘：m = ρ * π * r^2）
    mass = density * np.pi * radii**2
    return positions, velocities, radii, mass