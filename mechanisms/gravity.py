import numpy as np


def apply_gravity(positions, velocities, mass, dt, g=9.81):
    """
    Apply uniform gravity (free fall in -y).

    :param positions: (N, 2) positions
    :param velocities: (N, 2) velocities
    :param mass: (N,) mass (unused for g, kept for API symmetry)
    :param dt: time step [s]
    :param g: gravitational acceleration [m/s^2]
    :return: updated positions, velocities
    """
    velocities = velocities.copy()
    velocities[:, 1] -= g * dt
    positions = positions + velocities * dt
    return positions, velocities
