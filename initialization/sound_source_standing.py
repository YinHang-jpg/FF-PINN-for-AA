# sound_source_standing.py
import numpy as np

frequency = 10000
amplitude = 200

def compute_sound_field(domain_size=(0.1, 0.1), resolution=(200, 200),
                        amplitude=amplitude, frequency=frequency, sound_speed=340, time=0.0):
    """
    计算给定时间下的平面驻波声压场（x方向驻波）
    :param domain_size: 模拟区域尺寸（米）
    :param resolution: 网格点数（x方向, y方向）
    :param amplitude: 声压振幅（Pa）
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

    # 物理真实的驻波：p(x, t) = 2A cos(kx) cos(ωt)
    p = amplitude * 2 * np.cos(k * X) * np.cos(omega * time)
    return X, Y, p 