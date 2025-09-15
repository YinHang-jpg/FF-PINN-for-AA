import numpy as np

def cunningham_correction_factor(d_p, lambda_g):
    """
    计算Cunningham滑移修正因子
    正确公式：C_c = 1 + [2.514 + 0.800 exp(-0.550 d_p/λ_g)] · (λ_g / d_p)

    :param d_p: 粒子直径 (m)
    :param lambda_g: 气体分子平均自由程 (m)
    :return: Cunningham修正因子 (与粒径同形状)
    """
    d_p = np.asarray(d_p)
    eps = 1e-30
    ratio = np.clip(d_p / (lambda_g + eps), eps, None)
    exp_term = np.exp(-0.550 * ratio)
    bracket_term = 2.514 + 0.800 * exp_term
    C_c = 1.0 + bracket_term / ratio  # 等价于 bracket_term * (lambda_g / d_p)
    return C_c

def apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1.79e-5,
                      fluid_velocity=None, lambda_g=6.5e-8):
    """
    对粒子施加包含Cunningham滑移修正的斯托克斯阻力。

    F_drag = -3π μ_g d_p (V - U) / C_c

    :param positions: (N,2) 粒子位置（不直接用于拖曳计算，保留接口一致性）
    :param velocities: (N,2) 粒子速度
    :param radii: (N,) 粒子半径
    :param mass: (N,) 粒子质量
    :param dt: 时间步长
    :param viscosity: 流体动力粘度 (Pa·s)
    :param fluid_velocity: (N,2) 流体速度场，None则为零场
    :param lambda_g: 气体分子平均自由程 (m)
    :return: 更新后的 positions, velocities
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    radii = np.asarray(radii)
    mass = np.asarray(mass)
    N = positions.shape[0]

    if fluid_velocity is None:
        fluid_velocity = np.zeros((N, 2), dtype=velocities.dtype)

    d_p = 2.0 * radii  # (N,)
    C_c = cunningham_correction_factor(d_p, lambda_g)  # (N,)
    # 斯托克斯系数（逐粒子），再按分量乘以相对速度
    drag_coeff = (3.0 * np.pi * viscosity * d_p / C_c)[:, None]  # (N,1)
    relative_velocity = velocities - fluid_velocity  # (N,2)
    drag_force = -drag_coeff * relative_velocity  # (N,2)

    drag_acceleration = drag_force / mass[:, None]
    velocities = velocities + drag_acceleration * dt
    positions = positions + velocities * dt

    return positions, velocities
