# test_particle.py
import numpy as np

def initialize_test_particle(domain_size=(0.01, 0.01), speed=0.01, diameter=1e-6, density=1000):
    """
    初始化一个从右到左以固定速度移动的测试粒子
    
    参数：
        domain_size: 模拟域尺寸 (Lx, Ly)，单位：米
        speed: 粒子移动速度，单位：m/s
        diameter: 粒子直径，单位：米
        density: 粒子密度，单位：kg/m^3
    
    返回：
        positions: 位置数组 (1, 2)
        velocities: 速度数组 (1, 2)
        radii: 半径数组 (1,)
        mass: 质量数组 (1,)
    """
    Lx, Ly = domain_size
    
    # 粒子初始位置：在域的右边界附近
    start_x = Lx - 0.001  # 距离右边界1mm
    start_y = Ly / 2 - 5e-4# 在y方向居中
    positions = np.array([[start_x, start_y]])
    
    # 粒子速度：向左移动
    velocities = np.array([[-speed, 0.0]])
    
    # 粒子半径
    radius = diameter / 2
    radii = np.array([radius])
    
    # 计算质量（二维近似为圆盘：m = ρ * π * r^2）
    mass = density * np.pi * radius**2
    mass = np.array([mass])
    
    return positions, velocities, radii, mass

def update_test_particle_position(positions, velocities, domain_size, dt):
    """
    更新测试粒子的位置
    
    参数：
        positions: 当前位置 (1, 2)
        velocities: 当前速度 (1, 2)
        domain_size: 模拟域尺寸 (Lx, Ly)
        dt: 时间步长
    
    返回：
        更新后的位置 (1, 2)
    """
    Lx, Ly = domain_size
    
    # 更新位置
    new_positions = positions + velocities * dt
    
    # 边界处理：如果粒子到达左边界，重新从右边开始
    if new_positions[0, 0] <= 0.001:  # 留一点边距
        new_positions[0, 0] = Lx - 0.001  # 重新从右边开始
    
    # 确保y方向也在边界内
    if new_positions[0, 1] <= 0.001:
        new_positions[0, 1] = 0.001
    elif new_positions[0, 1] >= Ly - 0.001:
        new_positions[0, 1] = Ly - 0.001
    
    return new_positions

def get_test_particle_info():
    """
    获取测试粒子的信息
    
    返回：
        包含测试粒子信息的字典
    """
    return {
        'name': 'Test Particle',
        'description': 'A particle moving from right to left at constant speed',
        'color': 'red',
        'marker': 'o',
        'size': 100
    } 