import numpy as np

def apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1.79e-5):
    """
    对粒子施加斯托克斯阻力。
    :param positions: (N,2) 粒子位置
    :param velocities: (N,2) 粒子速度
    :param radii: (N,) 粒子半径
    :param mass: (N,) 粒子质量
    :param dt: 时间步长
    :param viscosity: 流体动力粘度 (Pa·s)
    :return: 更新后的 positions, velocities
    """
    # F_drag = -6 * pi * eta * r * v
    drag_coeff = 6 * np.pi * viscosity * radii  # (N,)
    # a_drag = F_drag / m = -drag_coeff * v / m
    a_drag = - (drag_coeff[:, None] * velocities) / mass[:, None]
    velocities = velocities + a_drag * dt
    positions = positions + velocities * dt
    return positions, velocities
