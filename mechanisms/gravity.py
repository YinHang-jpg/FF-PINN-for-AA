import numpy as np

def apply_gravity(positions, velocities, mass, dt, g=9.81):
    """
    对粒子施加重力，使其匀加速下落。
    :param positions: (N,2) 粒子位置
    :param velocities: (N,2) 粒子速度
    :param mass: (N,) 粒子质量
    :param dt: 时间步长
    :param g: 重力加速度，单位 m/s^2
    :return: 更新后的 positions, velocities
    """
    velocities = velocities.copy()
    # F = m * g, a = F / m = g
    velocities[:, 1] -= g * dt  # 只在y方向
    positions = positions + velocities * dt
    return positions, velocities
