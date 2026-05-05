import numpy as np

def cunningham_correction_factor(d_p, lambda_g):
    """
    Cunningham slip correction factor:
    C_c = 1 + [2.514 + 0.800 exp(-0.550 d_p/λ_g)] · (λ_g / d_p)

    Args:
        d_p: particle diameter [m]
        lambda_g: gas mean free path [m]
    Returns:
        C_c, same shape as d_p
    """
    d_p = np.asarray(d_p)
    eps = 1e-30
    ratio = np.clip(d_p / (lambda_g + eps), eps, None)
    exp_term = np.exp(-0.550 * ratio)
    bracket_term = 2.514 + 0.800 * exp_term
    C_c = 1.0 + bracket_term / ratio
    return C_c

def compute_air_velocity_due_to_sound(positions, time, frequency=10000, 
                                     sound_pressure_level=168.5, sound_speed=340):
    """
    Acoustic Eulerian fluid velocity for the standing-wave closure used in drivers.

    u(x,t) = -2*pi*f*A*cos(2*pi*x/lambda)*sin(2*pi*f*t), with A from the same
    pressure-to-displacement scaling as in ARF.get_amplitude (up to project constants).

    Args:
        positions: (N, 2) particle coordinates [m]
        time: current time [s]
        frequency: Hz
        sound_pressure_level: dB re 20 µPa
        sound_speed: m/s
    Returns:
        air_velocity (N, 2) [m/s]
    """
    positions = np.asarray(positions)
    N = positions.shape[0]
    
    wavelength = sound_speed / frequency
    
    reference_pressure = 20e-6  # Pa
    sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
    rho_0 = 1.225
    c_0 = 340
    A = sound_pressure / (rho_0 * c_0 * 2 * np.pi * frequency)
    
    omega = 2 * np.pi * frequency
    k = 2 * np.pi / wavelength
    
    x_positions = positions[:, 0]
    air_velocity_x = -2 * np.pi * frequency * A * np.cos(k * x_positions) * np.sin(omega * time)
    
    air_velocity = np.zeros((N, 2), dtype=positions.dtype)
    air_velocity[:, 0] = air_velocity_x
    air_velocity[:, 1] = 0.0
    
    return air_velocity

def apply_stokes_drag(positions, velocities, radii, mass, dt, viscosity=1.8e-5,
                      fluid_velocity=None, lambda_g=6.5e-8, time=0.0, 
                      use_sound_velocity=True, frequency=10000, sound_pressure_level=140,
                      fixed_diameter=2e-6, fixed_cunningham=1.083):
    """
    Stokes drag with fixed diameter / Cunningham (project defaults in several drivers).

    F = -3*pi*mu*d_p*(v - u_fluid) / C_c  (here d_p and C_c are the fixed constants above).

    Args:
        positions, velocities, radii, mass: particle arrays (radii unused; diameter fixed)
        dt: unused (forces only)
        fluid_velocity: optional (N, 2); if None, uses acoustic u or zeros
        time, frequency, sound_pressure_level: passed to compute_air_velocity_due_to_sound when needed

    Returns:
        drag_force (N, 2) [N]; velocities are not integrated here.
    """
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    N = positions.shape[0]

    if fluid_velocity is None:
        if use_sound_velocity:
            fluid_velocity = compute_air_velocity_due_to_sound(
                positions, time, frequency, sound_pressure_level
            )
        else:
            fluid_velocity = np.zeros((N, 2), dtype=velocities.dtype)
    else:
        fluid_velocity = np.asarray(fluid_velocity)

    drag_coeff = 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham
    relative_velocity = velocities - fluid_velocity
    drag_force = -drag_coeff * relative_velocity

    return drag_force
