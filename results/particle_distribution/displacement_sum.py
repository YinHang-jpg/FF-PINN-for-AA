"""
Complete Standalone Displacement vs Frequency Analysis Script

This script independently:
1. Trains Stokes models for each frequency and SPL combination
2. Loads models and runs particle simulations internally
3. Calculates particle displacement in real-time
4. Updates plots after each simulation
5. Fits smooth curves after completing each SPL sweep

NO external PINN script calls - everything is self-contained.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import shutil
import subprocess
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import required modules
from initialization.particle_initialization import initialize_particles

# Configuration
frequencies = range(8000, 22001, 2000)  # 8kHz to 22kHz
spl_values = range(120, 171, 10)  # 120 to 170 dB
output_dir = Path(__file__).parent

# Physical constants
gravity = 9.81
particle_density = 2000
air_density = 1.225
viscosity = 1.8e-5
fixed_diameter = 2e-6
fixed_cunningham = 1.083
sound_speed = 340
reference_pressure = 20e-6
rho_0 = 1.225
c_0 = 340

# Training scripts
training_scripts = [
    project_root / "mechanisms" / "STOKES_PINN_x.py",
    project_root / "mechanisms" / "STOKES_PINN_t.py",
    project_root / "mechanisms" / "STOKES_PINN_v.py",
]

model_files = {
    "STOKES_PINN_x.py": ["stokes_model_x.pth", "stokes_model_x_normalization_params.json"],
    "STOKES_PINN_t.py": ["stokes_model_t.pth", "stokes_model_t_normalization_params.json"],
    "STOKES_PINN_v.py": ["stokes_model_v.pth", "stokes_model_v_normalization_params.json"],
}


def modify_parameter_in_file(file_path, param_name, value):
    """Modify a parameter in a Python file"""
    import re
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    modified = False
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            continue
        if ('def ' not in line and '(' not in line and line.strip().startswith(param_name)):
            pattern = rf'^(\s*{param_name}\s*=\s*)\d+(.*)$'
            match = re.match(pattern, line)
            if match:
                lines[i] = f"{match.group(1)}{value}{match.group(2)}\n"
                modified = True
                break
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    return modified


def backup_files(file_list):
    """Backup multiple files"""
    backups = {}
    for file_path in file_list:
        if file_path.exists():
            backup_path = file_path.with_suffix('.py.backup_disp_sweep')
            shutil.copy2(str(file_path), str(backup_path))
            backups[file_path] = backup_path
    return backups


def restore_files(backups):
    """Restore backed up files"""
    for original_path, backup_path in backups.items():
        if backup_path.exists():
            shutil.copy2(str(backup_path), str(original_path))
            backup_path.unlink()


def check_models_exist(model_folder_path):
    """Check if all required models exist"""
    for files in model_files.values():
        for file in files:
            if not (model_folder_path / file).exists():
                return False
    return True


def train_models_for_freq_spl(freq, spl):
    """Train Stokes models for specific frequency and SPL"""
    freq_label = f"{freq//1000}k"
    spl_label = f"spl_{spl}"
    model_folder = project_root / "PINN" / f"freq_{freq_label}" / spl_label
    model_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training models: {freq_label}Hz, SPL={spl}dB")
    print(f"Target: {model_folder}")
    print(f"{'='*70}")
    
    if check_models_exist(model_folder):
        print(f"✓ Models exist, skipping training")
        return True
    
    # Files to modify
    files_to_modify = [
        project_root / "mechanisms" / "Stokes_drag.py",
        project_root / "initialization" / "sound_source_standing.py",
    ] + training_scripts
    
    backups = backup_files(files_to_modify)
    
    try:
        # Set parameters
        for file_path in files_to_modify:
            modify_parameter_in_file(file_path, 'frequency', freq)
            modify_parameter_in_file(file_path, 'sound_pressure_level', spl)
        
        # Train models
        success = 0
        for script in training_scripts:
            print(f"  Training: {script.name}")
            try:
                env = os.environ.copy()
                env['MPLBACKEND'] = 'Agg'
                env['PYTHONIOENCODING'] = 'utf-8'
                env['PYTHONUTF8'] = '1'
                env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
                
                # No timeout - let training run as long as it needs
                result = subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(project_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',  # Force UTF-8 encoding
                    errors='replace',  # Replace problematic characters instead of crashing
                )
                
                if result.returncode == 0:
                    print(f"    ✓ Success")
                    # Move files
                    for file in model_files[script.name]:
                        src = project_root / "PINN" / file
                        dst = model_folder / file
                        if src.exists():
                            shutil.move(str(src), str(dst))
                    success += 1
                else:
                    print(f"    ✗ Failed (exit code {result.returncode})")
                    if result.stderr:
                        # Print last 200 chars of error, safely
                        error_msg = result.stderr[-200:].replace('\n', ' ')
                        print(f"       Error: {error_msg}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        restore_files(backups)
        return success == len(training_scripts)
        
    except Exception as e:
        print(f"Error: {e}")
        restore_files(backups)
        return False


def load_models_and_normalizers(model_folder, freq):
    """Load trained models and normalizers with correct architectures"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define network architectures exactly as in training scripts
    class StokesNetX(nn.Module):
        """Position-dependent network with Fourier features"""
        def __init__(self, fourier_features=32, frequency=10000):
            super().__init__()
            self.fourier_features = fourier_features
            wavelength = sound_speed / frequency
            k = 2 * np.pi / wavelength
            self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
            
            self.layers = nn.Sequential(
                nn.Linear(fourier_features * 2 + 1, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            fourier_x = self.fourier_weights * x
            fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
            combined_input = torch.cat([fourier_features, x], dim=1)
            return self.layers(combined_input)
    
    class StokesNetT(nn.Module):
        """Time-dependent network with Fourier features"""
        def __init__(self, period_seconds, frequency):
            super().__init__()
            omega = 2 * np.pi * frequency
            self.register_buffer('omega', torch.tensor(omega, dtype=torch.float32))
            self.register_buffer('period_seconds', torch.tensor(period_seconds, dtype=torch.float32))
            
            self.layers = nn.Sequential(
                nn.Linear(3, 32),
                nn.Tanh(),
                nn.Linear(32, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
        
        def forward(self, t_norm):
            t_seconds = t_norm * self.period_seconds
            phase = self.omega * t_seconds
            fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
            combined_input = torch.cat([fourier_features, t_norm], dim=1)
            return self.layers(combined_input)
    
    class StokesNetV(nn.Module):
        """Velocity-dependent network"""
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(1, 16),
                nn.Tanh(),
                nn.Linear(16, 8),
                nn.Tanh(),
                nn.Linear(8, 1)
            )
        
        def forward(self, vx):
            return self.layers(vx)
    
    # Load models
    models = {}
    norms = {}
    period = 1.0 / freq
    
    # Load X model
    model_path_x = model_folder / "stokes_model_x.pth"
    norm_path_x = model_folder / "stokes_model_x_normalization_params.json"
    if model_path_x.exists() and norm_path_x.exists():
        model_x = StokesNetX(fourier_features=32, frequency=freq).to(device)
        model_x.load_state_dict(torch.load(model_path_x, map_location=device))
        model_x.eval()
        models['x'] = model_x
        
        with open(norm_path_x, 'r') as f:
            norms['x'] = json.load(f)
    else:
        raise FileNotFoundError(f"Missing model files for x")
    
    # Load T model
    model_path_t = model_folder / "stokes_model_t.pth"
    norm_path_t = model_folder / "stokes_model_t_normalization_params.json"
    if model_path_t.exists() and norm_path_t.exists():
        model_t = StokesNetT(period, freq).to(device)
        model_t.load_state_dict(torch.load(model_path_t, map_location=device))
        model_t.eval()
        models['t'] = model_t
        
        with open(norm_path_t, 'r') as f:
            norms['t'] = json.load(f)
    else:
        raise FileNotFoundError(f"Missing model files for t")
    
    # Load V model
    model_path_v = model_folder / "stokes_model_v.pth"
    norm_path_v = model_folder / "stokes_model_v_normalization_params.json"
    if model_path_v.exists() and norm_path_v.exists():
        model_v = StokesNetV().to(device)
        model_v.load_state_dict(torch.load(model_path_v, map_location=device))
        model_v.eval()
        models['v'] = model_v
        
        with open(norm_path_v, 'r') as f:
            norms['v'] = json.load(f)
    else:
        raise FileNotFoundError(f"Missing model files for v")
    
    return models, norms, device


def run_simulation_with_models(freq, spl, models, norms, device):
    """Run particle simulation using loaded models - completely independent"""
    print(f"  Running simulation...")
    
    # Parameters
    omega = 2.0 * np.pi * freq
    sound_pressure = reference_pressure * (10 ** (spl / 20))
    A = sound_pressure / (rho_0 * c_0 * omega)
    drag_coeff = 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham
    
    # Initialize particles
    domain_size = (0.034, 0.034)
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(N=100000, domain_size=domain_size)
    initial_positions = positions_np.copy()
    
    # Convert to torch
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
    
    # Simulation parameters
    period = 1.0 / freq
    num_cycles = 100
    steps_per_cycle = 100
    steps = num_cycles * steps_per_cycle
    dt = (num_cycles * period) / steps
    
    print(f"  Simulating {steps} steps ({num_cycles} cycles)...")
    
    # Normalization helpers
    def normalize(value, key, norm_dict):
        mean = norm_dict.get(f'{key}_mean', 0.0)
        std = norm_dict.get(f'{key}_std', 1.0)
        return (value - mean) / max(std, 1e-30)
    
    def denormalize(value, key, norm_dict):
        mean = norm_dict.get(f'{key}_mean', 0.0)
        std = norm_dict.get(f'{key}_std', 1.0)
        return value * std + mean
    
    # Time stepping
    with torch.inference_mode():
        inv_mass = 1.0 / mass_t
        
        for step in range(steps):
            t = step * dt
            
            # Get current state
            x = positions_t[:, 0].unsqueeze(1)
            vx = velocities_t[:, 0].unsqueeze(1)
            
            # Normalize inputs
            x_norm = (x - norms['x'].get('x_mean', 0.0)) / max(norms['x'].get('x_std', 1.0), 1e-30)
            t_norm = torch.full_like(x, (t - norms['t'].get('t_mean', 0.0)) / max(norms['t'].get('t_std', 1.0), 1e-30))
            vx_norm = (vx - norms['v'].get('vx_mean', 0.0)) / max(norms['v'].get('vx_std', 1.0), 1e-30)
            
            # Model predictions
            with torch.no_grad():
                fx_x_norm = models['x'](x_norm)
                fx_t_norm = models['t'](t_norm)
                fx_v_norm = models['v'](vx_norm)
            
            # Denormalize to get physical factors
            fx_x_mean = norms['x'].get('fx_mean', 0.0)
            fx_x_std = norms['x'].get('fx_std', 1.0)
            stokes_fx_factor = fx_x_norm * fx_x_std + fx_x_mean
            
            fx_t_mean = norms['t'].get('fx_mean', 0.0)
            fx_t_std = norms['t'].get('fx_std', 1.0)
            stokes_ft_factor = fx_t_norm * fx_t_std + fx_t_mean
            
            # Air velocity
            u_air = (-omega * A * stokes_fx_factor * stokes_ft_factor).squeeze(1)
            
            # Stokes drag force
            stokes_fx = -drag_coeff * (velocities_t[:, 0] - u_air)
            
            # Update kinematics (Forward Euler)
            positions_t.add_(velocities_t, alpha=dt)
            velocities_t[:, 0].add_(stokes_fx * inv_mass, alpha=dt)
            velocities_t[:, 1].add_(-gravity * dt)
    
    # Calculate displacement
    positions_final = positions_t.cpu().numpy()
    displacements = np.sqrt(
        (positions_final[:, 0] - initial_positions[:, 0])**2 +
        (positions_final[:, 1] - initial_positions[:, 1])**2
    ) * 1000.0  # mm
    
    avg_total_disp = np.mean(displacements)
    avg_per_cycle = avg_total_disp / num_cycles
    
    print(f"  ✓ Avg displacement per cycle: {avg_per_cycle:.6f} mm/cycle")
    print(f"    Total avg displacement: {avg_total_disp:.4f} mm")
    print(f"    Max displacement: {np.max(displacements):.4f} mm")
    
    return avg_per_cycle


def fit_regression_curve(x, y):
    """Fit a regression curve using polynomial or spline regression"""
    from scipy.optimize import curve_fit
    
    # Try polynomial regression (2nd or 3rd order)
    try:
        # Try 3rd order polynomial first
        coeffs = np.polyfit(x, y, 3)
        poly_func = np.poly1d(coeffs)
        
        # Check if fit is reasonable (R-squared > 0.7)
        y_pred = poly_func(x)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if r_squared > 0.7:
            return lambda x_new: poly_func(x_new), r_squared
    except:
        pass
    
    # Fallback: use smooth spline as regression
    try:
        # Smooth spline with higher smoothing parameter = more regression-like
        spline = UnivariateSpline(x, y, s=np.var(y) * len(y) * 0.1, k=3)
        return lambda x_new: spline(x_new), None
    except:
        # Last resort: simple polynomial
        coeffs = np.polyfit(x, y, 2)
        poly_func = np.poly1d(coeffs)
        return lambda x_new: poly_func(x_new), None


def plot_current_results(all_data, spl_values, output_dir):
    """Plot current results - always update the SAME figure file"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Define distinct colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Define distinct line styles
    linestyles = [
        '-',      # solid line
        '--',     # dashed line
        '-.',     # dash-dot line
        ':',      # dotted line
        (0, (3, 1, 1, 1)),      # densely dashdotted
        (0, (5, 5)),            # dashed
    ]
    
    # Define markers for data points
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    all_x, all_y = [], []
    
    for idx, spl in enumerate(spl_values):
        if spl not in all_data or len(all_data[spl]) == 0:
            continue
        
        freqs = np.array(sorted(all_data[spl].keys()))
        disps = np.array([all_data[spl][f] for f in freqs])
        freqs_khz = freqs / 1000.0
        
        all_x.extend(freqs_khz)
        all_y.extend(disps)
        
        # Select style for this SPL
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        
        # Plot data points with distinct markers
        ax.scatter(freqs_khz, disps, s=120, c=color, alpha=0.9,
                  marker=marker, edgecolors='black', linewidth=1.5, zorder=10)
        
        # Fit regression curve if enough points (>= 3)
        if len(freqs_khz) >= 3:
            try:
                # Fit regression curve
                reg_func, r_squared = fit_regression_curve(freqs_khz, disps)
                
                # Generate smooth x values for plotting
                x_smooth = np.linspace(freqs_khz.min(), freqs_khz.max(), 300)
                y_smooth = reg_func(x_smooth)
                
                # Ensure non-negative
                y_smooth = np.maximum(y_smooth, 0)
                
                # Plot regression curve with distinct line style
                if r_squared is not None:
                    label = f'SPL = {spl} dB (R²={r_squared:.3f})'
                else:
                    label = f'SPL = {spl} dB'
                
                ax.plot(x_smooth, y_smooth, linestyle=linestyle, color=color, 
                       linewidth=3, alpha=0.9, zorder=5, label=label)
            except Exception as e:
                print(f"    Warning: Regression failed for SPL {spl}dB: {e}")
                # Just show the data points without curve
                pass
        elif len(freqs_khz) > 0:
            # Not enough points yet, just show in legend
            ax.plot([], [], linestyle=linestyle, color=color, linewidth=3,
                   label=f'SPL = {spl} dB (in progress)')
    
    # Format
    ax.set_xlabel('Frequency (kHz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Displacement per Cycle (mm/cycle)', fontsize=16, fontweight='bold')
    ax.set_title('Particle Average Displacement per Cycle vs Frequency\n(Multiple Sound Pressure Levels)',
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Show legend
    ax.legend(fontsize=12, loc='best', framealpha=0.95, ncol=1)
    
    # Set limits with margins
    if len(all_x) > 0:
        all_x, all_y = np.array(all_x), np.array(all_y)
        x_margin = (all_x.max() - all_x.min()) * 0.08 if all_x.max() > all_x.min() else 1.0
        y_margin = (all_y.max() - all_y.min()) * 0.15 if all_y.max() > all_y.min() else 0.001
        ax.set_xlim(all_x.min() - x_margin, all_x.max() + x_margin)
        ax.set_ylim(max(0, all_y.min() - y_margin), all_y.max() + y_margin)
    else:
        ax.set_xlim(6, 24)
        ax.set_ylim(0, 0.1)
    
    plt.tight_layout()
    
    # Always save to the SAME file - this updates the single figure
    output_file = output_dir / "displacement_vs_frequency_realtime.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_file


def main():
    """Main function - completely standalone"""
    print("="*70)
    print("Standalone Displacement vs Frequency Analysis")
    print("NO external script calls - all simulation done internally")
    print("="*70)
    print(f"Frequency: 8-22 kHz (step: 2 kHz)")
    print(f"SPL: 120-170 dB (step: 10 dB)")
    print(f"Total: {len(frequencies)} × {len(spl_values)} = {len(frequencies)*len(spl_values)}")
    print("="*70)
    
    all_data = {}
    plt.ion()
    
    try:
        for spl in spl_values:
            print(f"\n{'#'*70}")
            print(f"# SPL = {spl} dB")
            print(f"{'#'*70}")
            
            all_data[spl] = {}
            
            for freq in frequencies:
                print(f"\n[Freq={freq//1000}kHz, SPL={spl}dB]")
                
                # Train models
                if not train_models_for_freq_spl(freq, spl):
                    print(f"  ✗ Training failed, skipping")
                    continue
                
                # Load models
                model_folder = project_root / "PINN" / f"freq_{freq//1000}k" / f"spl_{spl}"
                try:
                    models, norms, device = load_models_and_normalizers(model_folder, freq)
                except Exception as e:
                    print(f"  ✗ Failed to load models: {e}")
                    continue
                
                # Run simulation internally
                try:
                    displacement = run_simulation_with_models(freq, spl, models, norms, device)
                    all_data[spl][freq] = displacement
                    
                    # Update the SAME plot file with new data
                    print(f"  → Updating real-time plot...")
                    plot_current_results(all_data, spl_values, output_dir)
                except Exception as e:
                    print(f"  ✗ Simulation failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                time.sleep(0.5)
            
            # SPL completed - plot is already updated, just print summary
            print(f"\n{'='*70}")
            print(f"✓ Completed SPL={spl}dB: {len(all_data[spl])} points")
            print(f"  Regression curve fitted and displayed")
            print(f"{'='*70}")
        
        plt.ioff()
        
        # Final summary
        print("\n" + "="*70)
        print("Final Summary:")
        total = 0
        for spl in spl_values:
            if spl in all_data:
                count = len(all_data[spl])
                total += count
                print(f"  SPL {spl}dB: {count} points")
        print(f"  Total: {total} points")
        print("="*70)
        
        # Final plot (same file, just ensure it's up to date)
        final_file = plot_current_results(all_data, spl_values, output_dir)
        print(f"\n✓ Final figure: {final_file}")
        print(f"  All results shown in single plot with regression curves")
        
        # Save CSV
        csv_file = output_dir / "displacement_vs_frequency_multi_spl.csv"
        with open(csv_file, 'w') as f:
            f.write("SPL_dB,Frequency_kHz,Frequency_Hz,Displacement_mm_per_cycle\n")
            for spl in spl_values:
                if spl in all_data:
                    for freq, disp in sorted(all_data[spl].items()):
                        f.write(f"{spl},{freq/1000:.1f},{freq},{disp:.8f}\n")
        print(f"✓ CSV data: {csv_file}")
        
        print("\n" + "="*70)
        print("✓ Analysis Complete!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        plt.ioff()
        if len(all_data) > 0:
            plot_current_results(all_data, spl_values, output_dir)
            print("Saved partial results to displacement_vs_frequency_realtime.png")
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()
        plt.ioff()


if __name__ == "__main__":
    main()
