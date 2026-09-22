# Fix Windows console UTF-8
import sys
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# Import particle / sound initialization
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency as _frequency

# Import unified PINN stack
from mechanisms.UNIFIED_PINN import (
    load_individual_models,
    UnifiedNormalizer,
    PhysicalUnifiedModel,
)
 
# Physical parameters
frequency = 10000  # Hz (10 kHz)
gravity = 9.81  # Gravity acceleration (m/s²)


def calculate_concentration_distribution(positions, x_grid_mm):
    """Calculate concentration distribution using Gaussian kernel density estimation"""
    if len(positions) == 0:
        return np.zeros_like(x_grid_mm)
   
    # Convert particle x coordinates from meters to millimeters
    x_positions_mm = positions[:, 0] * 1000.0
   
    # Calculate concentration at each x_grid position
    concentrations = np.zeros_like(x_grid_mm)
   
    # Gaussian kernel density estimation parameter
    sigma = 0.2  # Standard deviation of concentration distribution (mm)
   
    for i, x_pos in enumerate(x_grid_mm):
        # Calculate distance of all particles to current x position
        distances = np.abs(x_positions_mm - x_pos)
       
        # Concentration calculation: closer distance, higher concentration
        # Use Gaussian function form for concentration distribution
        concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
       
        concentrations[i] = concentration
   
    return concentrations

def calculate_displacement_distribution(initial_pos, final_pos, x_grid_mm):
    """Calculate average displacement magnitude at each x position"""
    if len(initial_pos) == 0:
        return np.zeros_like(x_grid_mm)
    
    # Calculate total displacement magnitude for each particle (Euclidean distance)
    displacements = np.sqrt(
        (final_pos[:, 0] - initial_pos[:, 0])**2 + 
        (final_pos[:, 1] - initial_pos[:, 1])**2
    ) * 1000.0  # Convert to millimeters
    
    # Get initial x coordinates (millimeters)
    x_initial_mm = initial_pos[:, 0] * 1000.0
    
    # Calculate average displacement at each x_grid position for nearby particles
    avg_displacements = np.zeros_like(x_grid_mm)
    sigma = 0.5  # Kernel width for weighted average (mm)
    
    for i, x_pos in enumerate(x_grid_mm):
        # Calculate distance of all particles' initial positions to current x position
        distances = np.abs(x_initial_mm - x_pos)
        
        # Calculate weighted average displacement using Gaussian weights
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        if weights.sum() > 0:
            avg_displacements[i] = np.sum(weights * displacements) / weights.sum()
        else:
            avg_displacements[i] = 0.0
    
    return avg_displacements
 
 
def save_positions_to_csv(position_data, output_path, num_particles=None):
    """
    Save particle positions to CSV file in COMSOL format
    
    Args:
        position_data: Dict with 'times' (array) and 'positions' (list of arrays)
        output_path: Path to save CSV file
        num_particles: Number of particles to save (None = all particles)
    
    CSV Format:
        - Row 1: time_0, x_pos_particle1, x_pos_particle2, ...
        - Row 2: time_1, x_pos_particle1, x_pos_particle2, ...
        - ...
    """
    times = position_data['times']
    positions_list = position_data['positions']
    
    # Get total number of particles
    total_particles = len(positions_list[0])
    
    # Determine number of particles to save
    if num_particles is None:
        num_particles = total_particles
        print(f"\nSaving ALL particle positions to CSV...")
    else:
        num_particles = min(num_particles, total_particles)
        print(f"\nSaving subset of particle positions to CSV...")
        print(f"  Total particles: {total_particles}")
        print(f"  Saving: {num_particles}")
    
    print(f"  Time points: {len(times)}")
    print(f"  Output: {output_path}")
    
    # Prepare data matrix: each row is [time, pos1, pos2, ...]
    data_rows = []
    for t_idx, t_val in enumerate(times):
        row = [t_val] + positions_list[t_idx][:num_particles].tolist()
        data_rows.append(row)
    
    # Create DataFrame without header
    df = pd.DataFrame(data_rows)
    
    # Save to CSV without header (matching comsol_positions.csv format)
    df.to_csv(output_path, index=False, header=False)
    
    print(f"✓ Positions saved successfully!")
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Data shape: {len(times)} time points × {num_particles} particles")


def load_physical_unified_model(device):
    """Load physical unified model
   
    Args:
        device: torch device
    """
    #
    model_base_path = 'PINN'
   
    print(f"\n{'='*70}")
    print(f"Loading physical unified model...")
    print(f"Model path: {model_base_path}")
    print(f"{'='*70}\n")
   
    models = load_individual_models(device, model_base_path)
    # Unified normalizer aggregates normalization ranges/statistics of each sub-model
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': os.path.join(model_base_path, 'arf_model_x_normalization_params.json'),
        'arf_t_norm': os.path.join(model_base_path, 'arf_model_t_normalization_params.json'),
        'stokes_x_norm': os.path.join(model_base_path, 'stokes_model_x_normalization_params.json'),
        'stokes_t_norm': os.path.join(model_base_path, 'stokes_model_t_normalization_params.json'),
        'stokes_v_norm': os.path.join(model_base_path, 'stokes_model_v_normalization_params.json'),
    }
    unified_norm.load_from_individual_models(model_paths, device='cpu')
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
    return phys_model

def compute_force_using_physical_model(positions_t, velocities_t, t_scalar, phys_model, use_cuda: bool = False):
    """Calculate and return Fx (1D vector).
    For performance, only return x-direction force; y-direction gravity is directly applied as constant acceleration in main loop.
    """
    with torch.inference_mode():
        x = positions_t[:, 0]
        vx = velocities_t[:, 0]
        t_vec = torch.full_like(x, float(t_scalar))
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fx = phys_model(x, vx, t_vec)
        else:
            fx = phys_model(x, vx, t_vec)
        return fx.float() if fx.dtype != torch.float32 else fx
 
def run_single_simulation(total_steps, suppress_output=True, save_positions=False, position_interval=0.01):
    """Run single simulation and return results
   
    Args:
        total_steps: Total simulation steps
        suppress_output: Whether to suppress output
        save_positions: Whether to save particle positions at regular intervals
        position_interval: Time interval (in seconds) for saving positions
   
    Returns:
        tuple: (x_grid, relative_change) for subsequent analysis
               If save_positions=True, also returns position_data dict
    """
    # Active checkpoints live under PINN/
    model_base_path = 'PINN'
    
    if not suppress_output:
        print(f"Using model: {model_base_path}")
   
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Initialize particles
    domain_size = (0.034, 0.034)
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=4400, domain_size=domain_size)
    initial_positions = positions_np.copy()

    # Convert to torch tensors
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
   
    # Load model
    phys_model = load_physical_unified_model(device)
   
    # Fixed timestep parameters (no longer frequency-dependent)
    dt = 1e-6  # Fixed timestep: 1 microsecond
    steps = total_steps
    max_simulation_time = steps * dt
   
    if not suppress_output:
        print(f"Starting calculation: {steps} steps")
        print(f"Fixed timestep: {dt:.2e} s (0.1 microsecond)")
        print(f"Total simulation time: {max_simulation_time:.6f} s ({max_simulation_time*1e3:.3f} ms)")
        if save_positions:
            print(f"Position saving enabled: every {position_interval} s")
   
    total_start_time = time.perf_counter()
    
    # Initialize position tracking if requested
    position_records = []
    time_records = []
    next_save_time = 0.0  # First save at t=0
   
    # Time advancement (Forward Euler method)
    with torch.inference_mode():
        inv_mass_1d = 1.0 / mass_t
        simulation_time = 0.0
        
        for step in range(steps):
            # Save positions at specified intervals (check with small tolerance for floating point)
            if save_positions and abs(simulation_time - next_save_time) < dt/2:
                # Get current positions on CPU (convert to mm)
                current_positions_np = positions_t.detach().cpu().numpy()
                x_positions_mm = current_positions_np[:, 0] * 1000.0  # Convert to mm
                
                position_records.append(x_positions_mm.copy())
                time_records.append(simulation_time)
                
                if not suppress_output:
                    print(f"  Saved positions at t={simulation_time:.4f}s (step {step})")
                
                next_save_time += position_interval
            
            fx_current = compute_force_using_physical_model(positions_t, velocities_t, simulation_time, phys_model, use_cuda)
            positions_t.add_(velocities_t, alpha=dt)
            velocities_t[:, 0].add_(fx_current.mul(inv_mass_1d), alpha=dt)
            if gravity > 0:
                velocities_t[:, 1].add_(-gravity * dt)
            simulation_time += dt
        
        # Save final position if we haven't saved it yet (handle last time point)
        if save_positions:
            final_sim_time = steps * dt
            # Check if we need to save the final position
            if len(time_records) == 0 or abs(time_records[-1] - final_sim_time) > dt:
                current_positions_np = positions_t.detach().cpu().numpy()
                x_positions_mm = current_positions_np[:, 0] * 1000.0
                position_records.append(x_positions_mm.copy())
                time_records.append(final_sim_time)
                if not suppress_output:
                    print(f"  Saved final positions at t={final_sim_time:.4f}s (step {steps})")
   
    total_elapsed = time.perf_counter() - total_start_time
   
    if not suppress_output:
        print(f"Calculation completed: {total_elapsed:.2f}s")
        if save_positions:
            print(f"Total position snapshots saved: {len(time_records)}")
   
    # Calculate distribution
    positions_final_np = positions_t.detach().cpu().numpy()
    x_grid = np.linspace(2.0, 32.0, 100)
    
    initial_concentrations = calculate_concentration_distribution(initial_positions, x_grid)
    final_concentrations = calculate_concentration_distribution(positions_final_np, x_grid)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
   
    if save_positions:
        position_data = {
            'times': np.array(time_records),
            'positions': position_records  # List of arrays
        }
        return x_grid, relative_change, position_data
    else:
        return x_grid, relative_change

def main():
    """Main function - Run simulation with multiple timestep configurations and plot comparison
    
    Position saving is now integrated: saves positions first, then generates plots
    """
    # ==================== Configuration ====================
    SAVE_POSITIONS = True  # Set to True to save particle positions (runs before plotting)
    POSITION_INTERVAL = 0.01  # Save positions every 0.01 seconds
    POSITION_SAVE_STEPS = 100000  # Number of steps for position saving (0.1s total)
    NUM_PARTICLES_TO_SAVE = None  # Number of particles to save (None = all particles)
    
    # Prepare output directory
    output_dir = os.path.join('results', 'PD_vs_time')
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Step 1: Position Saving (if enabled) ====================
    if SAVE_POSITIONS:
        print("=" * 80)
        print("STEP 1: POSITION SAVING")
        print(f"Saving particle positions every {POSITION_INTERVAL}s")
        print(f"Total steps: {POSITION_SAVE_STEPS} (simulation time: {POSITION_SAVE_STEPS*1e-6}s)")
        if NUM_PARTICLES_TO_SAVE is None:
            print(f"Particles to save: ALL particles (100,000)")
        else:
            print(f"Particles to save: {NUM_PARTICLES_TO_SAVE}")
        print("=" * 80)
        
        # Run simulation with position tracking
        start_time = time.perf_counter()
        x_grid, relative_change, position_data = run_single_simulation(
            total_steps=POSITION_SAVE_STEPS, 
            suppress_output=False,
            save_positions=True,
            position_interval=POSITION_INTERVAL
        )
        elapsed_time = time.perf_counter() - start_time
        
        print(f"\n{'='*80}")
        print(f"Simulation completed in {elapsed_time:.2f}s")
        print(f"{'='*80}")
        
        # Save positions to CSV
        csv_output_path = os.path.join(output_dir, 'pinn_particle_positions.csv')
        
        save_positions_to_csv(position_data, csv_output_path, num_particles=NUM_PARTICLES_TO_SAVE)
        
        print(f"\n{'='*80}")
        print("✓ Position saving completed!")
        print(f"CSV file: {csv_output_path}")
        print(f"Time range: {position_data['times'][0]:.4f}s to {position_data['times'][-1]:.4f}s")
        print(f"Total snapshots: {len(position_data['times'])}")
        print(f"{'='*80}\n")
    
    # ==================== Step 2: Density Distribution Plotting ====================
    print("\n" + "=" * 80)
    print("STEP 2: DENSITY DISTRIBUTION PLOTTING")
    print("=" * 80)
    
    # Define timestep configurations - split into two groups
    # Group 1: 1e4, 3e4, 5e4 steps
    # Group 2: 6e4, 8e4, 1e5 steps
    group1_configs = [10000, 30000, 50000]
    group2_configs = [60000, 80000, 100000]
    all_timestep_configs = group1_configs + group2_configs
    
    print("Starting Multi-Timestep Comparison Simulation")
    print("Fixed timestep: 1e-6 s (1 microsecond)")
    print(f"Group 1: {group1_configs} steps")
    print(f"Group 2: {group2_configs} steps")
    print("=" * 80)
    
    # Use Agg backend for non-interactive plotting (no window popup)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Define colors and line styles - using different line types for each curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    linestyles = ['-', '--', '-.', '-', '--', '-.']  # solid, dashed, dash-dot
    linewidths = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    
    # Store all simulation results
    all_results = {}
    
    # Run simulation for each timestep configuration
    for idx, steps in enumerate(all_timestep_configs):
        print(f"\n{'='*80}")
        print(f"Running simulation {idx+1}/{len(all_timestep_configs)}: Total steps = {steps:,} ({steps:.1e})")
        print(f"Total simulation time: {steps * 1e-6:.6f} s ({steps * 1e-6 * 1e3:.3f} ms)")
        print(f"{'='*80}\n")
        
        start_time = time.perf_counter()
        x_grid, relative_change = run_single_simulation(total_steps=steps, suppress_output=False)
        elapsed_time = time.perf_counter() - start_time
        
        # Filter results: x < 2mm and x > 34mm
        valid_indices = (x_grid >= 2.0) & (x_grid <= 34.0)
        x_grid_filtered = x_grid[valid_indices]
        relative_change_filtered = relative_change[valid_indices]
        
        all_results[steps] = {
            'x_grid': x_grid_filtered,
            'relative_change': relative_change_filtered,
            'time': elapsed_time
        }
        
        print(f"\nSimulation completed! Time elapsed: {elapsed_time:.2f} seconds")
        print(f"Number of data points: {len(x_grid_filtered)}")
    
    # ========== Generate Figure 1: Group 1 (1e4, 3e4, 5e4) ==========
    print(f"\n{'='*80}")
    print("Generating Figure 1: Group 1 (1e4, 3e4, 5e4 steps)")
    print(f"{'='*80}")
    
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    for i, steps in enumerate(group1_configs):
        result = all_results[steps]
        x_grid_plot = result['x_grid']
        relative_change_plot = result['relative_change']
        color = colors[i]
        linestyle = linestyles[i]
        linewidth = linewidths[i]
        
        # Plot original data without smoothing
        sim_time_ms = steps * 1e-6 * 1e3
        ax1.plot(x_grid_plot, relative_change_plot, color=color, linestyle=linestyle,
               linewidth=linewidth, label=f'{steps:.1e} steps ({sim_time_ms:.3f}ms, {result["time"]:.1f}s)',
               alpha=0.85)
    
    # Set chart properties for Figure 1
    ax1.set_xlim(2.0, 34.0)
    ax1.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Relative Concentration Change (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Particle Density Distribution - Group 1 (dt=1μs)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9, title='Configuration (sim_time, calc_time)')
    
    # Dynamically adjust y-axis range
    all_y_data = []
    for line in ax1.get_lines():
        y_data = line.get_ydata()
        if len(y_data) > 0:
            all_y_data.extend(y_data)
    
    if len(all_y_data) > 0 and np.any(np.isfinite(all_y_data)):
        y_min = float(np.nanmin(all_y_data))
        y_max = float(np.nanmax(all_y_data))
        y_range = y_max - y_min
        y_margin = max(y_range * 0.15, 0.5)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # Save Figure 1
    output_path1 = os.path.join(output_dir, 'particle_distribution_group1.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved to: {output_path1}")
    plt.close(fig1)
    
    # ========== Generate Figure 2: Group 2 (6e4, 8e4, 1e5) ==========
    print(f"\n{'='*80}")
    print("Generating Figure 2: Group 2 (6e4, 8e4, 1e5 steps)")
    print(f"{'='*80}")
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    for i, steps in enumerate(group2_configs):
        result = all_results[steps]
        x_grid_plot = result['x_grid']
        relative_change_plot = result['relative_change']
        color = colors[i + 3]  # Use colors 3, 4, 5
        linestyle = linestyles[i + 3]
        linewidth = linewidths[i + 3]
        
        # Plot original data without smoothing
        sim_time_ms = steps * 1e-6 * 1e3
        ax2.plot(x_grid_plot, relative_change_plot, color=color, linestyle=linestyle,
               linewidth=linewidth, label=f'{steps:.1e} steps ({sim_time_ms:.3f}ms, {result["time"]:.1f}s)',
               alpha=0.85)
    
    # Set chart properties for Figure 2
    ax2.set_xlim(2.0, 34.0)
    ax2.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Relative Concentration Change (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Particle Density Distribution - Group 2 (dt=1μs)', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9, title='Configuration (sim_time, calc_time)')
    
    # Dynamically adjust y-axis range
    all_y_data = []
    for line in ax2.get_lines():
        y_data = line.get_ydata()
        if len(y_data) > 0:
            all_y_data.extend(y_data)
    
    if len(all_y_data) > 0 and np.any(np.isfinite(all_y_data)):
        y_min = float(np.nanmin(all_y_data))
        y_max = float(np.nanmax(all_y_data))
        y_range = y_max - y_min
        y_margin = max(y_range * 0.15, 0.5)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # Save Figure 2
    output_path2 = os.path.join(output_dir, 'particle_distribution_group2.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Figure 2 saved to: {output_path2}")
    plt.close(fig2)
    
    # Final processing after all simulations are complete
    print(f"\n{'='*80}")
    print("All simulations completed!")
    print(f"{'='*80}\n")
    
    # Save data to CSV file
    csv_path = os.path.join(output_dir, 'particle_distribution_data.csv')
    with open(csv_path, 'w') as f:
        # Write header
        f.write('x_mm')
        for steps in all_timestep_configs:
            f.write(f',relative_change_{steps:.1e}')
        f.write('\n')
        
        # Write data (using first configuration's x_grid as baseline)
        base_x_grid = all_results[all_timestep_configs[0]]['x_grid']
        for i, x_val in enumerate(base_x_grid):
            f.write(f'{x_val:.6f}')
            for steps in all_timestep_configs:
                result = all_results[steps]
                # Find closest x value
                idx_closest = np.argmin(np.abs(result['x_grid'] - x_val))
                f.write(f',{result["relative_change"][idx_closest]:.6f}')
            f.write('\n')
    
    print(f"Data saved to: {csv_path}")
    print(f"Figure 1 (Group 1) saved to: {output_path1}")
    print(f"Figure 2 (Group 2) saved to: {output_path2}")
 
if __name__ == "__main__":
    main()
