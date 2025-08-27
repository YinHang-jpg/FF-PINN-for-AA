import numpy as np

def compute_brownian_force(radii, temperature=293.15, fluid_viscosity=1.8e-5, dt=1e-3):
    """
    计算布朗力（随机力）
    
    根据爱因斯坦-斯托克斯关系计算扩散系数：
    D = k_B * T / (6 * π * η * r)
    
    布朗力的方差：
    <F²> = 2 * k_B * T * γ / dt
    
    其中 γ = 6 * π * η * r 是摩擦系数
    
    :param radii: 粒子半径数组 (N,)
    :param temperature: 温度，单位：K (默认293.15K = 20°C)
    :param fluid_viscosity: 流体粘度，单位：Pa·s (默认空气粘度1.8e-5)
    :param dt: 时间步长，单位：s
    :return: 布朗力数组 (N, 2)
    """
    # 玻尔兹曼常数
    k_B = 1.380649e-23  # J/K
    
    # 计算每个粒子的摩擦系数
    gamma = 6 * np.pi * fluid_viscosity * radii  # 摩擦系数
    
    # 计算布朗力的标准差
    # σ_F = sqrt(2 * k_B * T * γ / dt)
    force_std = np.sqrt(2 * k_B * temperature * gamma / dt)
    
    # 生成随机布朗力（高斯分布）
    N = len(radii)
    brownian_force = np.random.normal(0, force_std[:, np.newaxis], size=(N, 2))
    
    return brownian_force

def apply_brownian_motion(positions, velocities, radii, mass, dt, 
                         temperature=293.15, fluid_viscosity=1.8e-5):
    """
    应用布朗运动到粒子系统
    
    使用朗之万方程：
    m * dv/dt = F_drag + F_brownian
    
    其中：
    - F_drag = -γ * v (阻力)
    - F_brownian (随机力)
    
    :param positions: 粒子位置数组 (N, 2)
    :param velocities: 粒子速度数组 (N, 2)
    :param radii: 粒子半径数组 (N,)
    :param mass: 粒子质量数组 (N,)
    :param dt: 时间步长，单位：s
    :param temperature: 温度，单位：K
    :param fluid_viscosity: 流体粘度，单位：Pa·s
    :return: 更新后的位置和速度数组
    """
    # 计算摩擦系数
    gamma = 6 * np.pi * fluid_viscosity * radii  # 摩擦系数
    
    # 计算布朗力
    brownian_force = compute_brownian_force(radii, temperature, fluid_viscosity, dt)
    
    # 计算阻力
    drag_force = -gamma[:, np.newaxis] * velocities
    
    # 总力
    total_force = drag_force + brownian_force
    
    # 更新速度（使用欧拉方法）
    # dv = F * dt / m
    velocity_change = total_force * dt / mass[:, np.newaxis]
    velocities += velocity_change
    
    # 更新位置
    # dx = v * dt
    positions += velocities * dt
    
    return positions, velocities

def compute_diffusion_coefficient(radii, temperature=293.15, fluid_viscosity=1.8e-5):
    """
    计算扩散系数（爱因斯坦-斯托克斯关系）
    
    D = k_B * T / (6 * π * η * r)
    
    :param radii: 粒子半径数组 (N,)
    :param temperature: 温度，单位：K
    :param fluid_viscosity: 流体粘度，单位：Pa·s
    :return: 扩散系数数组 (N,)
    """
    # 玻尔兹曼常数
    k_B = 1.380649e-23  # J/K
    
    # 爱因斯坦-斯托克斯关系
    diffusion_coefficient = k_B * temperature / (6 * np.pi * fluid_viscosity * radii)
    
    return diffusion_coefficient

def apply_brownian_motion_simple(positions, velocities, radii, mass, dt,
                                temperature=293.15, fluid_viscosity=1.8e-5):
    """
    简化的布朗运动实现（直接添加随机位移）
    
    使用爱因斯坦关系：
    <x²> = 2 * D * t
    
    因此随机位移的标准差：
    σ_x = sqrt(2 * D * dt)
    
    :param positions: 粒子位置数组 (N, 2)
    :param velocities: 粒子速度数组 (N, 2)
    :param radii: 粒子半径数组 (N,)
    :param mass: 粒子质量数组 (N,)
    :param dt: 时间步长，单位：s
    :param temperature: 温度，单位：K
    :param fluid_viscosity: 流体粘度，单位：Pa·s
    :return: 更新后的位置和速度数组
    """
    # 计算扩散系数
    D = compute_diffusion_coefficient(radii, temperature, fluid_viscosity)
    
    # 计算随机位移的标准差
    displacement_std = np.sqrt(2 * D * dt)
    
    # 生成随机位移
    N = len(radii)
    random_displacement = np.random.normal(0, displacement_std[:, np.newaxis], size=(N, 2))
    
    # 应用随机位移
    positions += random_displacement
    
    # 更新速度（基于位移变化）
    velocities = random_displacement / dt
    
    return positions, velocities
