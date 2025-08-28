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
    # 均匀分布粒子位置（网格中心点），尽量覆盖整个域
    nx = int(np.ceil(np.sqrt(N)))
    ny = int(np.ceil(N / nx))
    x_centers = (np.arange(nx) + 0.5) * (Lx / nx)
    y_centers = (np.arange(ny) + 0.5) * (Ly / ny)
    X, Y = np.meshgrid(x_centers, y_centers)
    positions_grid = np.vstack((X.ravel(), Y.ravel())).T
    positions = positions_grid[:N]

    velocities = np.zeros_like(positions)
    radii = np.ones(N) * (diameter / 2)
    # 计算质量（三维近似球体：m = ρ * 4/3 * π * r^3）
    mass = density * 4/3 * np.pi * radii**3
    return positions, velocities, radii, mass