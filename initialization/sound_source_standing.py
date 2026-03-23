# sound_source_standing.py
import numpy as np

frequency = 22000  # Hz (10 kHz)
sound_pressure_level = 168.5  # dB

def spl_to_pressure(spl, reference_pressure=20e-6):
    """
    将声压级(SPL)转换为声压
    :param spl: 声压级 (dB)
    :param reference_pressure: 参考声压 (Pa), 默认为20微帕
    :return: 声压 (Pa)
    """
    return reference_pressure * (10 ** (spl / 20))

def compute_sound_field(domain_size=(0.034, 0.034), resolution=(200, 200),
                        spl=sound_pressure_level, frequency=frequency, sound_speed=340, time=0.0):
    """
    计算给定时间下的平面驻波声压场（x方向驻波）
    :param domain_size: 模拟区域尺寸（米）
    :param resolution: 网格点数（x方向, y方向）
    :param spl: 声压级（dB）
    :param frequency: 声波频率（Hz）
    :param sound_speed: 声速（m/s）
    :param time: 当前时刻 t（秒）
    :return: X, Y 网格坐标，和对应的声压值 p(x,y,t)
    """
    Lx, Ly = domain_size
    Nx, Ny = resolution
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    omega = 2 * np.pi * frequency
    k = omega / sound_speed
    
    # 将声压级转换为声压
    sound_pressure = spl_to_pressure(spl)
    
    # 转换为位移幅值：A = sound_pressure / (ρ₀c₀ω)
    rho_0 = 1.225  # 空气密度 (kg/m³)
    c_0 = 340      # 声速 (m/s)
    amplitude = sound_pressure / (rho_0 * c_0 * omega)
    
    # 使用图片中的声压公式：p(x,t) = 2πAp₀γsin(2πx/λ)cos(2πft)/λ
    p_0 = 101325    # Pa (大气压)
    gamma = 1.4     # 空气绝热指数
    wavelength = c_0 / frequency
    p = 2 * 2 * np.pi * amplitude * p_0 * gamma * np.sin(2 * np.pi * X / wavelength) * np.cos(2 * np.pi * frequency * time) / wavelength
    return X, Y, p