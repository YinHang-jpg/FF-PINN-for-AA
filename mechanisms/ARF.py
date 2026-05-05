import numpy as np
from initialization.sound_source_standing import (
    sound_pressure_level,
    spl_to_pressure,
    compute_sound_field,
    frequency,
)


def bilinear_interpolate(grid, x, y, dx, dy):
    Nx = grid.shape[1]
    Ny = grid.shape[0]
    ix = np.clip(x / dx, 0, Nx - 2)
    iy = np.clip(y / dy, 0, Ny - 2)
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    x1 = np.clip(x0 + 1, 0, Nx - 1)
    y1 = np.clip(y0 + 1, 0, Ny - 1)
    wx = ix - x0
    wy = iy - y0
    # grid indexed as [row y, col x]
    return (
        (1 - wx) * (1 - wy) * grid[y0, x0]
        + wx * (1 - wy) * grid[y0, x1]
        + (1 - wx) * wy * grid[y1, x0]
        + wx * wy * grid[y1, x1]
    )


p_0 = 101325.0  # Pa
# Air (ideal gas constants for standing-wave model)
rho_0 = 1.225  # [kg/m^3]
gamma = 1.4
c_0 = 340.0  # [m/s]; frequency imported from sound_source_standing


def get_amplitude():
    """
    Pressure amplitude scale from SPL (implementation doubles SPL pressure once here).
    Returns displacement-related amplitude term used in the manuscript pressure form.
    """
    sound_pressure = 2 * spl_to_pressure(sound_pressure_level)
    A = sound_pressure / rho_0 / c_0 / 2 / np.pi / frequency
    return A


# Snapshot at import for plotting helpers (plot_x.py / plot_t.py)
amplitude = get_amplitude()


def get_particle_arf_force(particle_x, particle_y, particle_radius, time, domain_size=(0.034, 0.034)):
    """
    Acoustic radiation force on one spherical particle from front/back pressures on x-axis.

    Uses sampling points x ± sqrt(2) d_p / 4 on the pressure field.
    """
    d_p = 2 * particle_radius
    offset = np.sqrt(2) * d_p / 4

    x_front = particle_x - offset
    x_back = particle_x + offset

    _, _, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=time)

    Lx, Ly = domain_size
    x_grid = np.linspace(0, Lx, 200)

    p_front = np.interp(x_front, x_grid, P[0, :])
    p_back = np.interp(x_back, x_grid, P[0, :])

    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4
    arf_force = np.array([force_magnitude, 0.0])

    return arf_force


def compute_pressure_gradient_and_apply_arf(positions, radii, time):
    """
    ARF on each particle from analytical standing wave (same formula as grid-based path).

    F_p = π d_p^2 [p(x - sqrt(2) d_p/4, t) - p(x + sqrt(2) d_p/4, t)] / 4

    Returns:
        arf_force: (N, 2), x-component only
    """
    wavelength = c_0 / frequency
    arf_force = np.zeros_like(positions)
    for i, (x, y) in enumerate(positions):
        d_p = 2 * radii[i]
        offset = np.sqrt(2) * d_p / 4

        x_front = x - offset
        x_back = x + offset

        k = 2 * np.pi / wavelength
        omega = 2 * np.pi * frequency
        amplitude = get_amplitude()

        p_front = (
            2
            * 2
            * np.pi
            * amplitude
            * p_0
            * gamma
            * np.sin(k * x_front)
            * np.cos(omega * time)
            / wavelength
        )
        p_back = (
            2
            * 2
            * np.pi
            * amplitude
            * p_0
            * gamma
            * np.sin(k * x_back)
            * np.cos(omega * time)
            / wavelength
        )

        pressure_diff = p_front - p_back
        force_magnitude = np.pi * d_p**2 * pressure_diff / 4
        arf_force[i, 0] = force_magnitude
        arf_force[i, 1] = 0.0

    return arf_force
