# test_particle.py
import numpy as np

def initialize_test_particle(domain_size=(0.01, 0.01), speed=0.01, diameter=1e-6, density=1000):
    """
    Single test particle, sliding left at constant speed.

    Args:
        domain_size: (Lx, Ly) [m]
        speed: |vx| [m/s]
        diameter: [m]
        density: [kg/m^3]

    Returns:
        positions (1, 2), velocities (1, 2), radii (1,), mass (1,)
    """
    Lx, Ly = domain_size
    
    start_x = Lx - 0.001
    start_y = Ly / 2 - 5e-4
    positions = np.array([[start_x, start_y]])
    
    velocities = np.array([[-speed, 0.0]])
    
    radius = diameter / 2
    radii = np.array([radius])
    
    mass = density * np.pi * radius**2
    mass = np.array([mass])
    
    return positions, velocities, radii, mass

def update_test_particle_position(positions, velocities, domain_size, dt):
    """
    Euler step with periodic wrap when reaching the left pad.
    """
    Lx, Ly = domain_size
    
    new_positions = positions + velocities * dt
    
    if new_positions[0, 0] <= 0.001:
        new_positions[0, 0] = Lx - 0.001
    
    if new_positions[0, 1] <= 0.001:
        new_positions[0, 1] = 0.001
    elif new_positions[0, 1] >= Ly - 0.001:
        new_positions[0, 1] = Ly - 0.001
    
    return new_positions

def get_test_particle_info():
    return {
        'name': 'Test Particle',
        'description': 'A particle moving from right to left at constant speed',
        'color': 'red',
        'marker': 'o',
        'size': 100
    }
