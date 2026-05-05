# sound_source_standing.py
import numpy as np

frequency = 10000  # Hz
sound_pressure_level = 168.5  # dB


def spl_to_pressure(spl, reference_pressure=20e-6):
    """
    Convert sound pressure level (SPL) to pressure amplitude.

    :param spl: SPL [dB]
    :param reference_pressure: reference pressure [Pa], default 20 µPa
    :return: pressure [Pa]
    """
    return reference_pressure * (10 ** (spl / 20))


def compute_sound_field(domain_size=(0.034, 0.034), resolution=(200, 200),
                        spl=sound_pressure_level, frequency=frequency, sound_speed=340, time=0.0):
    """
    Standing plane wave along x: pressure field p(x, y, t).

    :param domain_size: domain (Lx, Ly) [m]
    :param resolution: grid size (Nx, Ny)
    :param spl: sound pressure level [dB]
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

    sound_pressure = spl_to_pressure(spl)

    rho_0 = 1.225  # air density [kg/m^3]
    c_0 = 340  # sound speed [m/s]
    amplitude = sound_pressure / (rho_0 * c_0 * omega)

    p_0 = 101325.0  # Pa, atmospheric pressure
    gamma = 1.4  # ratio of specific heats
    wavelength = c_0 / frequency
    p = (
        2 * 2 * np.pi * amplitude * p_0 * gamma
        * np.sin(2 * np.pi * X / wavelength)
        * np.cos(2 * np.pi * frequency * time)
        / wavelength
    )
    return X, Y, p
