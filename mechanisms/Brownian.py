import numpy as np

def compute_brownian_force(radii, temperature=293.15, fluid_viscosity=1.8e-5, dt=1e-3):
    """
    Gaussian stochastic force consistent with Einstein–Stokes fluctuation scaling.

    D = k_B T / (6 pi eta r); force variance ~ 2 k_B T gamma / dt with gamma = 6 pi eta r.

    Args:
        radii: (N,) [m]
        temperature: [K]
        fluid_viscosity: [Pa·s]
        dt: timestep [s]

    Returns:
        (N, 2) force samples [N]
    """
    k_B = 1.380649e-23  # J/K
    
    gamma = 6 * np.pi * fluid_viscosity * radii
    
    force_std = np.sqrt(2 * k_B * temperature * gamma / dt)
    
    N = len(radii)
    brownian_force = np.random.normal(0, force_std[:, np.newaxis], size=(N, 2))
    
    return brownian_force

def apply_brownian_motion(positions, velocities, radii, mass, dt, 
                         temperature=293.15, fluid_viscosity=1.8e-5):
    """
    Langevin-style Euler update with linear drag and thermal noise.

    m dv/dt = -gamma v + F_B; positions += v dt.
    """
    gamma = 6 * np.pi * fluid_viscosity * radii
    
    brownian_force = compute_brownian_force(radii, temperature, fluid_viscosity, dt)
    
    drag_force = -gamma[:, np.newaxis] * velocities
    
    total_force = drag_force + brownian_force
    
    velocity_change = total_force * dt / mass[:, np.newaxis]
    velocities += velocity_change
    
    positions += velocities * dt
    
    return positions, velocities

def compute_diffusion_coefficient(radii, temperature=293.15, fluid_viscosity=1.8e-5):
    """Einstein–Stokes diffusion coefficient k_B T / (6 pi eta r)."""
    k_B = 1.380649e-23  # J/K
    
    diffusion_coefficient = k_B * temperature / (6 * np.pi * fluid_viscosity * radii)
    
    return diffusion_coefficient

def apply_brownian_motion_simple(positions, velocities, radii, mass, dt,
                                temperature=293.15, fluid_viscosity=1.8e-5):
    """
    Direct displacement increment: x <- x + N(0, 2 D dt) in each axis; v <- dx/dt.
    """
    D = compute_diffusion_coefficient(radii, temperature, fluid_viscosity)
    
    displacement_std = np.sqrt(2 * D * dt)
    
    N = len(radii)
    random_displacement = np.random.normal(0, displacement_std[:, np.newaxis], size=(N, 2))
    
    positions += random_displacement
    
    velocities = random_displacement / dt
    
    return positions, velocities
