import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import time
from scipy.interpolate import PchipInterpolator

from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level, compute_sound_field
from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as ARFNormalizerX
from mechanisms.ARF_PINN_t import ARFNetT, Normalizer as ARFNormalizerT
from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

USE_STOKES_DRAG = True
USE_ARF = True
dt = 1e-6  # [s]
gravity = 9.81  # [m/s^2]
particle_density = 1000  # [kg/m^3]
air_density = 1.225  # [kg/m^3]
viscosity = 1.8e-5  # [Pa·s]
fixed_diameter = 2e-6  # [m]
fixed_cunningham = 1.083


def load_individual_models_and_norms():
    models = {}
    norms = {}
    # Checkpoint paths
    model_paths = {
        'arf_x': 'PINN/arf_model_x.pth',
        'arf_t': 'PINN/arf_model_t.pth',
        'stokes_x': 'PINN/stokes_model_x.pth',
        'stokes_t': 'PINN/stokes_model_t.pth',
        'stokes_v': 'PINN/stokes_model_v.pth',
    }
    norm_paths = {
        'arf_x': 'PINN/arf_model_x_normalization_params.json',
        'arf_t': 'PINN/arf_model_t_normalization_params.json',
        'stokes_x': 'PINN/stokes_model_x_normalization_params.json',
        'stokes_t': 'PINN/stokes_model_t_normalization_params.json',
        'stokes_v': 'PINN/stokes_model_v_normalization_params.json',
    }
    # Load ARF x
    if os.path.exists(model_paths['arf_x']) and os.path.exists(norm_paths['arf_x']):
        models['arf_x'] = ARFNetX(fourier_features=32)
        models['arf_x'].load_state_dict(torch.load(model_paths['arf_x'], map_location='cpu', weights_only=True))
        models['arf_x'].eval()
        norms['arf_x'] = ARFNormalizerX(); norms['arf_x'].load(norm_paths['arf_x'], device='cpu')
    else:
        return None, None
    # Load ARF t
    if os.path.exists(model_paths['arf_t']) and os.path.exists(norm_paths['arf_t']):
        models['arf_t'] = ARFNetT(period_seconds=1.0/frequency)
        models['arf_t'].load_state_dict(torch.load(model_paths['arf_t'], map_location='cpu', weights_only=True))
        models['arf_t'].eval()
        norms['arf_t'] = ARFNormalizerT(); norms['arf_t'].load(norm_paths['arf_t'], device='cpu')
    else:
        return None, None
    # Load Stokes x
    if os.path.exists(model_paths['stokes_x']) and os.path.exists(norm_paths['stokes_x']):
        models['stokes_x'] = StokesNetX(fourier_features=32)
        models['stokes_x'].load_state_dict(torch.load(model_paths['stokes_x'], map_location='cpu', weights_only=True))
        models['stokes_x'].eval()
        norms['stokes_x'] = StokesNormalizerX(); norms['stokes_x'].load(norm_paths['stokes_x'], device='cpu')
    else:
        return None, None
    # Load Stokes t
    if os.path.exists(model_paths['stokes_t']) and os.path.exists(norm_paths['stokes_t']):
        models['stokes_t'] = StokesNetT(period_seconds=1.0/frequency)
        models['stokes_t'].load_state_dict(torch.load(model_paths['stokes_t'], map_location='cpu', weights_only=True))
        models['stokes_t'].eval()
        norms['stokes_t'] = StokesNormalizerT(); norms['stokes_t'].load(norm_paths['stokes_t'], device='cpu')
    else:
        return None, None
    # Load Stokes v
    if os.path.exists(model_paths['stokes_v']) and os.path.exists(norm_paths['stokes_v']):
        models['stokes_v'] = StokesNetV()
        models['stokes_v'].load_state_dict(torch.load(model_paths['stokes_v'], map_location='cpu', weights_only=True))
        models['stokes_v'].eval()
        norms['stokes_v'] = StokesNormalizerV(); norms['stokes_v'].load(norm_paths['stokes_v'], device='cpu')
    else:
        return None, None
    return models, norms

def compute_force_using_trained_models(positions_t, velocities_t, t_scalar, models, norms):
    with torch.no_grad():
        # Run sub-networks on CPU to avoid VRAM spikes
        x = positions_t[:, 0].detach().cpu()
        vx = velocities_t[:, 0].detach().cpu()
        t_vec = torch.full_like(x, float(t_scalar))

        # Normalize inputs
        x_norm_arf = norms['arf_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_arf = norms['arf_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        x_norm_stokes = norms['stokes_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_stokes = norms['stokes_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        vx_norm_stokes = norms['stokes_v'].transform({'vx': vx.unsqueeze(1)})['vx'].squeeze()

        # Sub-model forward
        arf_fx_norm = models['arf_x'](x_norm_arf.unsqueeze(1))
        arf_ft_norm = models['arf_t'](t_norm_arf.unsqueeze(1))
        stokes_fx_norm = models['stokes_x'](x_norm_stokes.unsqueeze(1))
        stokes_ft_norm = models['stokes_t'](t_norm_stokes.unsqueeze(1))
        stokes_fv_norm = models['stokes_v'](vx_norm_stokes.unsqueeze(1))

        # Denormalize ARF branches
        arf_fx = norms['arf_x'].inverse({'fx': arf_fx_norm})['fx'].squeeze()
        arf_ft = norms['arf_t'].inverse({'fx': arf_ft_norm})['fx'].squeeze()

        # Denormalize Stokes factors
        # - stokes_x_factor ~ cos(kx)
        # - stokes_t_factor ~ cos(ωt)
        # - use physical vx for the velocity term to avoid inconsistent norm JSON units
        stokes_fx_factor = norms['stokes_x'].inverse({'fx': stokes_fx_norm})['fx'].squeeze()
        stokes_ft_factor = norms['stokes_t'].inverse({'fx': stokes_ft_norm})['fx'].squeeze()
        stokes_v_value = vx

        # Physical scaling (phase cos)
        drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
        sound_pressure = 20e-6 * (10 ** (sound_pressure_level / 20))
        A = sound_pressure / (1.225 * 340 * 2 * np.pi * 10000)
        stokes_amp = 2 * np.pi * 10000 * A

        # Total force (manuscript-style combination)
        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - stokes_amp * stokes_ft_factor * stokes_fx_factor)

        total_fx_cpu = arf_total + stokes_total
        fy = torch.zeros_like(total_fx_cpu)
        total_force = torch.stack([total_fx_cpu, fy], dim=1).to(positions_t.device)
        return total_force

def compute_force_components_using_trained_models(positions_t, velocities_t, t_scalar, models, norms):
    """Return ARF and Stokes components (1D CPU tensors, length N)."""
    with torch.no_grad():
        x = positions_t[:, 0].detach().cpu()
        vx = velocities_t[:, 0].detach().cpu()
        t_vec = torch.full_like(x, float(t_scalar))

        x_norm_arf = norms['arf_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_arf = norms['arf_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        x_norm_stokes = norms['stokes_x'].transform({'x': x.unsqueeze(1)})['x'].squeeze()
        t_norm_stokes = norms['stokes_t'].transform({'t': t_vec.unsqueeze(1)})['t'].squeeze()
        vx_norm_stokes = norms['stokes_v'].transform({'vx': vx.unsqueeze(1)})['vx'].squeeze()

        arf_fx_norm = models['arf_x'](x_norm_arf.unsqueeze(1))
        arf_ft_norm = models['arf_t'](t_norm_arf.unsqueeze(1))
        stokes_fx_norm = models['stokes_x'](x_norm_stokes.unsqueeze(1))
        stokes_ft_norm = models['stokes_t'](t_norm_stokes.unsqueeze(1))
        stokes_fv_norm = models['stokes_v'](vx_norm_stokes.unsqueeze(1))

        arf_fx = norms['arf_x'].inverse({'fx': arf_fx_norm})['fx'].squeeze()
        arf_ft = norms['arf_t'].inverse({'fx': arf_ft_norm})['fx'].squeeze()
        stokes_fx_factor = norms['stokes_x'].inverse({'fx': stokes_fx_norm})['fx'].squeeze()
        stokes_ft_factor = norms['stokes_t'].inverse({'fx': stokes_ft_norm})['fx'].squeeze()
        stokes_v_value = norms['stokes_v'].inverse({'fx': stokes_fv_norm})['fx'].squeeze()

        drag_coeff = 3.0 * np.pi * 1.8e-5 * 2e-6 / 1.083
        sound_pressure = 20e-6 * (10 ** (sound_pressure_level / 20))
        A = sound_pressure / (1.225 * 340 * 2 * np.pi * 10000)
        stokes_amp = 2 * np.pi * 10000 * A

        arf_total = arf_fx * arf_ft
        stokes_total = -drag_coeff * (stokes_v_value - stokes_amp * stokes_ft_factor * stokes_fx_factor)
        return arf_total, stokes_total

def compute_particle_forces_using_models(positions_t, velocities_t, mass_t, radii_t, dt, t_scalar, models, norms):
    """Advance one step using the five trained sub-models."""
    try:
        forces_t = compute_force_using_trained_models(positions_t, velocities_t, t_scalar, models, norms)
        accelerations_t = forces_t / mass_t[:, None]
        velocities_t = velocities_t + accelerations_t * dt
        positions_t = positions_t + velocities_t * dt
        return positions_t, velocities_t, (forces_t[:, 0], forces_t[:, 1])
    except Exception:
        return positions_t, velocities_t, None

def main():
    """Batch integration without live animation."""
    global positions, velocities, mass, radii, initial_positions, domain_size
    
    print("=== Unified PINN particle simulation (batch mode) ===")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if use_cuda:
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    domain_size = (0.034, 0.034)  # 34 mm square
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=10000, domain_size=domain_size)
    initial_positions = positions_np.copy()

    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    radii_t = torch.as_tensor(radii_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)

    ones_vec = torch.ones(positions_t.shape[0], device=device)
    
    print(f"Initialized {positions_t.shape[0]} particles")
    print(f"Domain: {domain_size[0]*1000:.1f} mm x {domain_size[1]*1000:.1f} mm")
    
    print("\nLoading trained factorized models (ARF / Stokes)...")
    models, norms = load_individual_models_and_norms()
    if models is None or norms is None:
        print("Error: missing checkpoint or normalization JSON under PINN/")
        return

    # torch.compile disabled (Triton); re-enable when the runtime supports it
    # import torch._dynamo
    # torch._dynamo.config.suppress_errors = True
    
    print("Factorized models loaded.")
    
    steps = 1000
    simulation_time = 0.0
    print(f"\nIntegrating: {steps} steps, dt = {dt:.2e} s")
    
    total_start_time = time.perf_counter()
    
    arf_init, stokes_init = compute_force_components_using_trained_models(positions_t, velocities_t, 0.0, models, norms)
    print("Forces at t = 0:")
    print(f"  ARF    range: [{arf_init.min():.2e}, {arf_init.max():.2e}] N | mean: {arf_init.mean():.2e} N")
    print(f"  Stokes range: [{stokes_init.min():.2e}, {stokes_init.max():.2e}] N | mean: {stokes_init.mean():.2e} N")
    for step in range(steps):
        simulation_time += dt
        t = simulation_time

        positions_before = positions_t.clone()
        velocities_before = velocities_t.clone()

        positions_t, velocities_t, forces = compute_particle_forces_using_models(
            positions_t, velocities_t, mass_t, radii_t, dt, t, models, norms
        )

    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    print(f"\n=== Done ===")
    print(f"Steps: {steps}")
    print(f"dt: {dt:.2e} s")
    print(f"Wall time: {total_elapsed:.4f} s ({total_elapsed/60:.2f} min)")
    print(f"Mean time per step: {total_elapsed/steps*1e3:.3f} ms")
    print(f"Final simulation time: {simulation_time:.6f} s")
    
    print("\nPlotting distributions...")
    
    positions_final_np = positions_t.detach().cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.scatter(initial_positions[:, 0] * 1000, initial_positions[:, 1] * 1000, 
               s=1, c='blue', alpha=0.5, label='Initial')
    ax1.scatter(positions_final_np[:, 0] * 1000, positions_final_np[:, 1] * 1000, 
               s=1, c='red', alpha=0.5, label='Final')
    ax1.set_xlim(0, domain_size[0] * 1000)
    ax1.set_ylim(0, domain_size[1] * 1000)
    ax1.set_xlabel('X Position (mm)')
    ax1.set_ylabel('Y Position (mm)')
    ax1.set_title('Particle Positions: Initial vs Final')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Gaussian KDE along x (same recipe as comsol_visualizer.py)
    x_grid = np.linspace(0.0, 34.0, 100)
    
    def calculate_concentration_distribution(positions, x_grid_mm):
        """Gaussian KDE along x (mm), matching comsol_visualizer recipe."""
        if len(positions) == 0:
            return np.zeros_like(x_grid_mm)
        
        x_positions_mm = positions[:, 0] * 1000.0
        
        concentrations = np.zeros_like(x_grid_mm)
        
        sigma = 0.2  # kernel std (mm)
        
        for i, x_pos in enumerate(x_grid_mm):
            distances = np.abs(x_positions_mm - x_pos)
            concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
            
            concentrations[i] = concentration
        
        return concentrations
    
    total_particles = len(initial_positions)
    initial_concentration = np.full_like(x_grid, total_particles / len(x_grid))
    
    final_concentration = calculate_concentration_distribution(positions_final_np, x_grid)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentration - initial_concentration) / initial_concentration * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n=== Concentration diagnostics ===")
    print(f"Initial (uniform): {initial_concentration[0]:.3f}")
    print(f"Final concentration range: {np.min(final_concentration):.3f} – {np.max(final_concentration):.3f}")
    print(f"Mean initial: {np.mean(initial_concentration):.3f}")
    print(f"Mean final: {np.mean(final_concentration):.3f}")
    print(f"Particle count: {total_particles}")
    print(f"Relative change (%): min {np.min(relative_change):.3f}, max {np.max(relative_change):.3f}")
    print(f"Mean relative change (%): {np.mean(relative_change):.3f}")
    
    print("Plotting relative change vs uniform baseline (percent).")
    display_data = relative_change
    ylabel = 'Relative Change (%)'
    
    ax2.set_xlim(0.0, 34.0)
    ax2.set_xlabel('X Position (mm)')
    ax2.set_ylabel(ylabel)
    ax2.set_title('Particle Density Distribution')
    ax2.grid(True, alpha=0.3)
    
    bars = ax2.bar(x_grid, display_data, width=0.34, align='center', alpha=0.8, 
                   color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration change')
    
    from scipy.interpolate import PchipInterpolator
    try:
        pchip = PchipInterpolator(x_grid, display_data, extrapolate=False)
        x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
        y_smooth = pchip(x_smooth)
        curve_line, = ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line, = ax2.plot(x_grid, display_data, 'r-', linewidth=2, label='Smoothed')
    
    if np.any(np.isfinite(display_data)):
        y_min = float(np.nanmin(display_data))
        y_max = float(np.nanmax(display_data))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Spatial statistics ===")
    print(f"Initial count: {len(initial_positions)}")
    print(f"Final count: {len(positions_final_np)}")
    print(f"Initial x (mm): {initial_positions[:, 0].min()*1000:.2f} – {initial_positions[:, 0].max()*1000:.2f}")
    print(f"Final x (mm): {positions_final_np[:, 0].min()*1000:.2f} – {positions_final_np[:, 0].max()*1000:.2f}")
    
    print("\nSimulation finished.")

if __name__ == "__main__":
    main()
