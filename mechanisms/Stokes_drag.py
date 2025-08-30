import numpy as np

def cunningham_correction_factor(d_p, lambda_g):
    """
    计算Cunningham滑移修正因子
    C_c = 1 + [2.514 + 0.800exp(-0.550d_p/λ_g)]λ_g/d_p
    
    :param d_p: 粒子直径 (m)
    :param lambda_g: 气体分子平均自由程 (m)
    :return: Cunningham修正因子
    """
    ratio = d_p / lambda_g
    exp_term = np.exp(-0.550 * ratio)
    bracket_term = 2.514 + 0.800 * exp_term
    C_c = 1 + bracket_term * ratio
    return C_c

def apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1.79e-5, 
                      fluid_velocity=None, lambda_g=6.5e-8):
    """
    对粒子施加包含Cunningham滑移修正的斯托克斯阻力。
    
    根据公式：F_x = F_p - 3πμ_g d_p (V_x - U_x - u) / C_c
    
    :param positions: (N,2) 粒子位置
    :param velocities: (N,2) 粒子速度
    :param radii: (N,) 粒子半径
    :param mass: (N,) 粒子质量
    :param dt: 时间步长
    :param viscosity: 流体动力粘度 (Pa·s)，默认空气粘度
    :param fluid_velocity: 流体速度场 (N,2)，如果为None则假设流体静止
    :param lambda_g: 气体分子平均自由程 (m)，默认空气在标准条件下的值
    :return: 更新后的 positions, velocities
    """
    N = len(positions)
    
    # 如果没有提供流体速度，假设流体静止
    if fluid_velocity is None:
        fluid_velocity = np.zeros((N, 2))
    
    # 计算每个粒子的阻力
    drag_force = np.zeros_like(positions)
    
    for i in range(N):
        # 粒子直径
        d_p = 2 * radii[i]
        
        # 计算Cunningham修正因子
        C_c = cunningham_correction_factor(d_p, lambda_g)
        
        # 相对速度：V_particle - U_fluid
        relative_velocity = velocities[i] - fluid_velocity[i]
        
        # 斯托克斯阻力：F_drag = -3πμ_g d_p (V - U) / C_c
        drag_coeff = 3 * np.pi * viscosity * d_p / C_c
        drag_force[i] = -drag_coeff * relative_velocity
    
    # 计算阻力加速度
    drag_acceleration = drag_force / mass[:, None]
    
    # 更新速度和位置（显式欧拉）
    velocities = velocities + drag_acceleration * dt
    positions = positions + velocities * dt
    
    return positions, velocities
