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

def compute_air_velocity_due_to_sound(positions, time, frequency=10000, 
                                     sound_pressure_level=140, sound_speed=340):
    """
    计算声波影响下的空气速度场。
    
    根据公式：u(x,t) = -2πfA cos(2πx/λ)sin(2πft)
    其中 A 通过 get_amplitude() 函数获得
    
    :param positions: (N,2) 粒子位置
    :param time: 当前时间 (s)
    :param frequency: 声波频率 (Hz)
    :param sound_pressure_level: 声压级 (dB)
    :param sound_speed: 声速 (m/s)
    :return: air_velocity (N,2) 空气速度向量
    """
    positions = np.asarray(positions)
    N = positions.shape[0]
    
    # 计算波长
    wavelength = sound_speed / frequency
    
    # 计算振幅 A（与ARF.py中的get_amplitude函数保持一致）
    reference_pressure = 20e-6  # Pa
    sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
    rho_0 = 1.225  # 空气密度 (kg/m³)
    c_0 = 340      # 声速 (m/s)
    A = sound_pressure / (rho_0 * c_0 * 2 * np.pi * frequency)
    
    # 计算空气速度：u(x,t) = -2πfA cos(2πx/λ)sin(2πft)
    omega = 2 * np.pi * frequency
    k = 2 * np.pi / wavelength
    
    # 只考虑x方向的空气速度（驻波沿x方向）
    x_positions = positions[:, 0]
    air_velocity_x = -2 * np.pi * frequency * A * np.cos(k * x_positions) * np.sin(omega * time)
    
    # y方向速度为0（驻波只在x方向）
    air_velocity = np.zeros((N, 2), dtype=positions.dtype)
    air_velocity[:, 0] = air_velocity_x
    air_velocity[:, 1] = 0.0
    
    return air_velocity

def apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1.8e-5,
                      fluid_velocity=None, lambda_g=6.5e-8, time=0.0, 
                      use_sound_velocity=True, frequency=10000, sound_pressure_level=140,
                      fixed_diameter=2e-6, fixed_cunningham=1.083):
    """
    对粒子施加包含Cunningham滑移修正的斯托克斯阻力。

    使用固定公式：F = -3*π*1.8e-5*2e-6*(v - 2*π*f*A*cos(2*π*x/λ)*sin(2*π*f*t))/1.083

    :param positions: (N,2) 粒子位置
    :param velocities: (N,2) 粒子速度
    :param radii: (N,) 粒子半径（在此函数中不使用，使用固定直径）
    :param mass: (N,) 粒子质量
    :param dt: 时间步长
    :param viscosity: 流体动力粘度 (Pa·s)，默认1.8e-5
    :param fluid_velocity: (N,2) 流体速度场，None则根据use_sound_velocity决定
    :param lambda_g: 气体分子平均自由程 (m)
    :param time: 当前时间 (s)，用于计算声波影响下的空气速度
    :param use_sound_velocity: 是否使用声波影响下的空气速度
    :param frequency: 声波频率 (Hz)
    :param sound_pressure_level: 声压级 (dB)
    :param fixed_diameter: 固定粒子直径 (m)，默认2e-6
    :param fixed_cunningham: 固定Cunningham修正因子，默认1.083
    :return: drag_force (N,2) 阻力向量（不在此处更新速度与位置）
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    N = positions.shape[0]

    # 计算流体速度
    if fluid_velocity is None:
        if use_sound_velocity:
            # 使用声波影响下的空气速度
            fluid_velocity = compute_air_velocity_due_to_sound(
                positions, time, frequency, sound_pressure_level
            )
        else:
            # 使用零速度场
            fluid_velocity = np.zeros((N, 2), dtype=velocities.dtype)
    else:
        # 使用提供的流体速度
        fluid_velocity = np.asarray(fluid_velocity)

    # 使用固定参数计算斯托克斯阻力
    # F = -3*π*μ*d_p*(v - u)/C_c
    # 其中 μ=1.8e-5, d_p=2e-6, C_c=1.083
    drag_coeff = 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham
    relative_velocity = velocities - fluid_velocity  # (N,2)
    drag_force = -drag_coeff * relative_velocity  # (N,2)

    return drag_force
