# sound_source_traveling.py
import numpy as np

frequency = 10000
amplitude = 2000


def compute_sound_field(domain_size=(0.01, 0.1), resolution=(200, 200),
                        amplitude=amplitude, frequency=frequency, sound_speed=343, time=0.0):
    """
    Traveling plane wave along +x: p(x, y, t).

    :param domain_size: domain (Lx, Ly) [m]
    :param resolution: grid size (Nx, Ny)
    :param amplitude: pressure amplitude [Pa]
    :param frequency: frequency [Hz]
    :param sound_speed: speed of sound [m/s]
    :param time: time t [s]
    :return: X, Y mesh grids and pressure p(x, y, t)
    """
    Lx, Ly = domain_size
    Nx, Ny = resolution
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    omega = 2 * np.pi * frequency
    k = omega / sound_speed

    p = amplitude * np.cos(k * X - omega * time)
    return X, Y, p
