"""
frequency - 
frequency()
 PINN_main_integrated.py 
"""

# Fix Windows console UTF-8
import sys
if sys.platform.startswith('win'):
    import codecs
    if hasattr(sys.stdout, 'detach'):
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except (AttributeError, OSError):
            pass

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path

# Repo root on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

#
from initialization.particle_initialization import initialize_particles
#
#

#
FREQUENCIES = [8000, 10000, 12000, 16000]

# Physical parameters
gravity = 9.81  # Gravity acceleration (m/s²)
SPL_DB = 168.5  # Fixed SPL level (dB)

#
N_PARTICLES = 44000  #  4400, 10 

#
FREQ_COLORS = {
    8000: '#1f77b4',   # 
    10000: '#ff7f0e',  # 
    12000: '#2ca02c',  # 
    16000: '#d62728',  # 
}

FREQ_LINESTYLES = {
    8000: '-',    # 
    10000: '--',  # 
    12000: ':',   # 
    16000: '-.',  # 
}

FREQ_MARKERS = {
    8000: 'o',   # 
    10000: 's',  # 
    12000: '^',  # 
    16000: 'd',  # 
}


def calculate_concentration_distribution(positions, x_grid_mm):
    """Calculate concentration distribution using Gaussian kernel density estimation"""
    if len(positions) == 0:
        return np.zeros_like(x_grid_mm)
   
    # Convert particle x coordinates from meters to millimeters
    x_positions_mm = positions[:, 0] * 1000.0
   
    # Calculate concentration at each x_grid position
    concentrations = np.zeros_like(x_grid_mm)
   
    # Gaussian kernel density estimation parameter
    sigma = 0.3  # Standard deviation of concentration distribution (mm)
   
    for i, x_pos in enumerate(x_grid_mm):
        # Calculate distance of all particles to current x position
        distances = np.abs(x_positions_mm - x_pos)
       
        # Concentration calculation: closer distance, higher concentration
        concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
       
        concentrations[i] = concentration
   
    return concentrations


def load_physical_unified_model(device, freq_folder):
    """Load physical unified model for specific frequency
   
    Args:
        device: torch device
        freq_folder: frequency folder name (e.g., 'freq_8k')
    """
    #
    import importlib
    import sys
    
    #
    modules_to_reload = [
        'mechanisms.UNIFIED_PINN',
        'mechanisms.ARF_PINN_x',
        'mechanisms.ARF_PINN_t',
        'mechanisms.STOKES_PINN_x',
        'mechanisms.STOKES_PINN_t',
        'mechanisms.STOKES_PINN_v',
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    #
    from mechanisms.UNIFIED_PINN import (
        load_individual_models,
        UnifiedNormalizer,
        PhysicalUnifiedModel,
    )
    
    model_base_path = os.path.join('PINN', freq_folder)
   
    print(f"\n{'='*70}")
    print(f"Loading physical unified model...")
    print(f"Model path: {model_base_path}")
    print(f"[]")
    
    #
    model_files = [
        'arf_model_x.pth',
        'arf_model_t.pth',
        'stokes_model_x.pth',
        'stokes_model_t.pth',
        'stokes_model_v.pth',
    ]
    
    missing_files = []
    for model_file in model_files:
        full_path = os.path.join(model_base_path, model_file)
        if not os.path.exists(full_path):
            missing_files.append(model_file)
    
    if missing_files:
        print(f"  ✗ Error: Missing model files in {model_base_path}:")
        for f in missing_files:
            print(f"    - {f}")
        raise FileNotFoundError(f"Missing model files: {missing_files}")
    else:
        print(f"  ✓ All model files found in {model_base_path}")
    
    print(f"{'='*70}\n")
   
    #
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
    import re
    m = re.search(r'freq_(\d+)k', str(freq_folder))
    pack_freq = int(m.group(1)) * 1000 if m else None
    phys_model = PhysicalUnifiedModel(
        models, unified_norm, device=device, frequency=pack_freq
    )
    
    print(f"  ✓ Physical unified model loaded successfully from {model_base_path}\n")
    
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


def run_simulation_for_frequency(frequency, num_cycles=1000, particle_density=2000, num_particles=None):
    """Run simulation for a specific frequency and return aggregation metric
   
    Args:
        frequency: Frequency in Hz
        num_cycles: Number of cycles to simulate (default: 1000)
        particle_density: Particle density (kg/m³)
        num_particles: Number of particles (default: N_PARTICLES from module)
   
    Returns:
        dict: Results dictionary with aggregation metrics
    """
    freq_k = frequency // 1000
    freq_folder = f'freq_{freq_k}k'
    
    print(f"\n{'='*80}")
    print(f"Processing frequency: {frequency} Hz ({freq_k}k)")
    print(f"SPL: {SPL_DB} dB (fixed)")
    print(f"Model folder: PINN/{freq_folder}/")
    print(f"{'='*80}")
    
    # Check if model folder exists
    model_path = os.path.join(project_root, 'PINN', freq_folder)
    if not os.path.exists(model_path):
        print(f"  ✗ Model folder does not exist: {model_path}")
        print(f"  : frequency {frequency}Hz PINN")
        return None
    
    print(f"  ✓ Model folder found: {model_path}")
    
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # Initialize particles with specified density
    n_particles = num_particles if num_particles is not None else N_PARTICLES
    domain_size = (0.034, 0.034)
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(
        N=n_particles, domain_size=domain_size, density=particle_density)
    initial_positions = positions_np.copy()

    # Convert to torch tensors
    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
   
    #
    print(f"\nfrequency {frequency}Hz ({freq_k}k) PINN...")
    phys_model = load_physical_unified_model(device, freq_folder)
   
    # Calculate timestep parameters based on number of cycles
    # Total simulation time = num_cycles * period = num_cycles / frequency
    period = 1.0 / frequency  # Period in seconds
    total_simulation_time = num_cycles * period  # Total time in seconds
    
    dt = 1e-6  # Fixed timestep: 1 microsecond
    steps = int(total_simulation_time / dt)  # Calculate required steps
   
    print(f"Starting calculation:")
    print(f"  - Number of cycles: {num_cycles}")
    print(f"  - Period: {period*1e6:.3f} μs")
    print(f"  - Total simulation time: {total_simulation_time:.6f} s ({total_simulation_time*1e3:.3f} ms)")
    print(f"  - Timestep: {dt:.2e} s (1 μs)")
    print(f"  - Total steps: {steps}")
   
    total_start_time = time.perf_counter()
   
    # Time advancement (Forward Euler method)
    from tqdm import tqdm

    with torch.inference_mode():
        inv_mass_1d = 1.0 / mass_t
        simulation_time = 0.0
        pbar = tqdm(
            range(steps),
            desc=f"{freq_k}kHz / {num_cycles}cyc",
            unit="step",
            mininterval=0.5,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for step in pbar:
            fx_current = compute_force_using_physical_model(
                positions_t, velocities_t, simulation_time, phys_model, use_cuda
            )
            positions_t.add_(velocities_t, alpha=dt)
            velocities_t[:, 0].add_(fx_current.mul(inv_mass_1d), alpha=dt)
            if gravity > 0:
                velocities_t[:, 1].add_(-gravity * dt)
            simulation_time += dt
            if step % 500 == 0 or step + 1 == steps:
                elapsed = time.perf_counter() - total_start_time
                rate = (step + 1) / max(elapsed, 1e-9)
                eta = (steps - step - 1) / max(rate, 1e-9)
                pbar.set_postfix(
                    t_us=f"{simulation_time*1e6:.1f}",
                    rate=f"{rate:.0f}/s",
                    ETA=f"{eta:.0f}s",
                    refresh=False,
                )
   
    total_elapsed = time.perf_counter() - total_start_time
    print(f"Calculation completed: {total_elapsed:.2f}s")
   
    # Calculate distribution
    positions_final_np = positions_t.detach().cpu().numpy()
    x_grid = np.linspace(2.0, 32.0, 500)
    
    initial_concentrations = calculate_concentration_distribution(initial_positions, x_grid)
    final_concentrations = calculate_concentration_distribution(positions_final_np, x_grid)
    
    # Calculate relative change
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
        relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate aggregation metrics
    # 1. Peak concentration increase (maximum relative change)
    peak_increase = np.max(relative_change)
    peak_position = x_grid[np.argmax(relative_change)]
    
    # 2. Standard deviation of concentration (higher = more aggregation)
    concentration_std = np.std(final_concentrations)
    
    # 3. Concentration coefficient of variation (CV)
    concentration_cv = concentration_std / np.mean(final_concentrations) if np.mean(final_concentrations) > 0 else 0
    
    print(f"\n=== Aggregation Metrics ===")
    print(f"Peak concentration increase: {peak_increase:.2f}% at x={peak_position:.2f}mm")
    print(f"Concentration std: {concentration_std:.4f}")
    print(f"Concentration CV: {concentration_cv:.4f}")
    
    results = {
        'frequency_hz': frequency,
        'frequency_khz': freq_k,
        'num_cycles': num_cycles,
        'num_particles': n_particles,
        'total_simulation_time': total_simulation_time,
        'peak_increase_percent': peak_increase,
        'peak_position_mm': peak_position,
        'concentration_std': concentration_std,
        'concentration_cv': concentration_cv,
        'computation_time_s': total_elapsed,
        'x_grid': x_grid,
        'relative_change': relative_change,
        'initial_concentrations': initial_concentrations,
        'final_concentrations': final_concentrations,
    }
    
    return results


def _draw_concentration_combined_plot(all_results, output_path, num_cycles_label):
    """frequency( concentration_distribution_combined )."""
    fig, ax = plt.subplots(figsize=(14, 8))
    for result in all_results:
        if result is not None:
            freq_hz = result['frequency_hz']
            freq_label = f"{result['frequency_khz']} kHz"
            color = FREQ_COLORS.get(freq_hz, 'gray')
            linestyle = FREQ_LINESTYLES.get(freq_hz, '-')
            marker = FREQ_MARKERS.get(freq_hz, 'o')
            ax.plot(result['x_grid'], result['relative_change'],
                   color=color, linestyle=linestyle, linewidth=5,
                   marker=marker, markevery=50, markersize=8, alpha=0.85,
                   label=freq_label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Position (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Concentration Change (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Concentration Distribution for Different Frequencies (SPL={SPL_DB}dB, {num_cycles_label})',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=13, framealpha=0.95, edgecolor='black', shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(all_results_100, all_results_1000, output_dir):
    """Save aggregation results to CSV and plots (100 cycles + 1000 cycles)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _to_csv_data(all_results):
        return [
            {
                'frequency_hz': r['frequency_hz'],
                'frequency_khz': r['frequency_khz'],
                'num_cycles': r['num_cycles'],
                'num_particles': r['num_particles'],
                'total_simulation_time_s': r['total_simulation_time'],
                'peak_increase_percent': r['peak_increase_percent'],
                'peak_position_mm': r['peak_position_mm'],
                'concentration_std': r['concentration_std'],
                'concentration_cv': r['concentration_cv'],
                'computation_time_s': r['computation_time_s'],
            }
            for r in all_results if r is not None
        ]

    csv_data_100 = _to_csv_data(all_results_100)
    csv_data_1000 = _to_csv_data(all_results_1000)

    if csv_data_100:
        df100 = pd.DataFrame(csv_data_100)
        df100.to_csv(output_dir / 'aggregation_metrics_100cycles.csv', index=False, encoding='utf-8')
        print(f"✓ Saved aggregation metrics (100 cycles) to: {output_dir / 'aggregation_metrics_100cycles.csv'}")
    if csv_data_1000:
        df1000 = pd.DataFrame(csv_data_1000)
        df1000.to_csv(output_dir / 'aggregation_metrics_1000cycles.csv', index=False, encoding='utf-8')
        print(f"✓ Saved aggregation metrics (1000 cycles) to: {output_dir / 'aggregation_metrics_1000cycles.csv'}")

    # Combined concentration plots (manuscript Fig. 10) + curve archive
    def _archive_curves(all_results, cycles_tag: int) -> None:
        payload = {}
        for r in all_results:
            if r is None:
                continue
            fk = int(r["frequency_khz"])
            payload[f"x_{cycles_tag}_{fk}"] = np.asarray(r["x_grid"], float)
            payload[f"y_{cycles_tag}_{fk}"] = np.asarray(r["relative_change"], float)
        return payload

    curve_payload = {}
    if any(r is not None for r in all_results_100):
        path_100 = output_dir / 'concentration_distribution_combined_100cycles.png'
        _draw_concentration_combined_plot(all_results_100, path_100, '100 cycles')
        print(f"✓ Saved combined concentration distribution (100 cycles) to: {path_100}")
        curve_payload.update(_archive_curves(all_results_100, 100))
    if any(r is not None for r in all_results_1000):
        path_1000 = output_dir / 'concentration_distribution_combined_1000cycles.png'
        _draw_concentration_combined_plot(all_results_1000, path_1000, '1000 cycles')
        print(f"✓ Saved combined concentration distribution (1000 cycles) to: {path_1000}")
        curve_payload.update(_archive_curves(all_results_1000, 1000))
    if curve_payload:
        npz_path = output_dir / "fig10_concentration_curves.npz"
        # Merge with any existing archive so a 100-only / 1000-only run
        # does not wipe the other cycle set.
        if npz_path.is_file():
            old = np.load(npz_path)
            merged = {k: old[k] for k in old.files}
            merged.update(curve_payload)
            curve_payload = merged
        np.savez_compressed(npz_path, **curve_payload)
        print(f"✓ Saved Fig. 10 curves to: {npz_path}")
        try:
            sys.path.insert(0, str(project_root / "results"))
            from _figures import FIGURES  # noqa: WPS433
            import shutil
            for cycles, src_name, results in (
                (100, "concentration_distribution_combined_100cycles.png", all_results_100),
                (1000, "concentration_distribution_combined_1000cycles.png", all_results_1000),
            ):
                src = output_dir / src_name
                if src.is_file() and any(r is not None for r in results):
                    dst = FIGURES / f"fig10_concentration_{cycles}cycles.png"
                    shutil.copy2(src, dst)
                    print(f"✓ Published manuscript Fig. 10 → {dst}")
        except Exception as exc:  # pragma: no cover
            print(f"⚠ Could not publish Fig. 10 to figures/: {exc}")

    #
    csv_data = csv_data_1000 if csv_data_1000 else csv_data_100
    if csv_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Particle Aggregation Metrics vs Frequency (SPL={SPL_DB}dB)', 
                     fontsize=16, fontweight='bold')
        
        frequencies = [r['frequency_khz'] for r in csv_data]
        
        # Plot 1: Peak increase
        axes[0, 0].plot(frequencies, [r['peak_increase_percent'] for r in csv_data], 'o-', linewidth=4, markersize=8)
        axes[0, 0].set_xlabel('Frequency (kHz)', fontsize=12)
        axes[0, 0].set_ylabel('Peak Increase (%)', fontsize=12)
        axes[0, 0].set_title('Peak Concentration Increase', fontsize=13, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Peak position
        axes[0, 1].plot(frequencies, [r['peak_position_mm'] for r in csv_data], 's-', linewidth=4, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Frequency (kHz)', fontsize=12)
        axes[0, 1].set_ylabel('Peak Position (mm)', fontsize=12)
        axes[0, 1].set_title('Position of Peak Concentration', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Concentration std
        axes[1, 0].plot(frequencies, [r['concentration_std'] for r in csv_data], '^-', linewidth=4, markersize=8, color='green')
        axes[1, 0].set_xlabel('Frequency (kHz)', fontsize=12)
        axes[1, 0].set_ylabel('Concentration Std', fontsize=12)
        axes[1, 0].set_title('Concentration Standard Deviation', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Concentration CV
        axes[1, 1].plot(frequencies, [r['concentration_cv'] for r in csv_data], 'd-', linewidth=4, markersize=8, color='red')
        axes[1, 1].set_xlabel('Frequency (kHz)', fontsize=12)
        axes[1, 1].set_ylabel('Concentration CV', fontsize=12)
        axes[1, 1].set_title('Concentration Coefficient of Variation', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'aggregation_metrics_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved aggregation metrics plot to: {plot_path}")


def main():
    """Main function"""
    import argparse
    ap = argparse.ArgumentParser(description="Frequency sweep for manuscript Fig. 10")
    ap.add_argument(
        "--cycles",
        type=int,
        choices=(100, 1000),
        default=None,
        help="Run only this cycle count (default: both 100 then 1000).",
    )
    args = ap.parse_args()

    print("="*80)
    print("frequency - ")
    print("="*80)
    print(f"\nSPL: {SPL_DB} dB")
    print(f"frequency: {[f'{f//1000}k' for f in FREQUENCIES]} Hz")
    print(f" {len(FREQUENCIES)} frequency")
    if args.cycles is None:
        cycle_plan = [(100, "all_results_100"), (1000, "all_results_1000")]
        print("Cycle sets: 100 and 1000")
    else:
        cycle_plan = [(args.cycles, f"all_results_{args.cycles}")]
        print(f"Cycle sets: {args.cycles} only")
    
    # Check if all model folders exist
    print("\n...")
    print(f"PINN: {project_root / 'PINN'}")
    
    #
    pinn_dir = project_root / 'PINN'
    if pinn_dir.exists():
        freq_folders = [d.name for d in pinn_dir.iterdir() if d.is_dir() and d.name.startswith('freq_')]
        if freq_folders:
            print(f"\nfrequency:")
            for folder in sorted(freq_folders):
                print(f"  - {folder}")
        else:
            print(f"\n⚠ : PINN freq_* ")
    
    print(f"\n...")
    missing_folders = []
    for freq in FREQUENCIES:
        freq_k = freq // 1000
        model_path = os.path.join(project_root, 'PINN', f'freq_{freq_k}k')
        if not os.path.exists(model_path):
            missing_folders.append(f'freq_{freq_k}k')
            print(f"  ✗ freq_{freq_k}k - ")
        else:
            print(f"  ✓ freq_{freq_k}k - ")
    
    if missing_folders:
        print(f"\n⚠ : :")
        for folder in missing_folders:
            print(f"  - PINN/{folder}")
        print("\nfrequencyPINN！")
        print("frequency,...")
    else:
        print("\n✓ ")
    
    print("\nStarting frequency sweep...")
    
    #
    all_results_100 = []
    all_results_1000 = []
    wall_start = time.perf_counter()
    case_total = len(FREQUENCIES) * len(cycle_plan)
    case_i = 0

    for num_cycles, _tag in cycle_plan:
        result_list = all_results_100 if num_cycles == 100 else all_results_1000
        print(f"\n\n{'@'*80}")
        print(f"  >>> {num_cycles} cycles <<<")
        print(f"{'@'*80}")
        for i, freq in enumerate(FREQUENCIES, 1):
            case_i += 1
            print(f"\n\n{'#'*80}")
            print(f"Case {case_i}/{case_total}: frequency {freq} Hz ({num_cycles} cycles)")
            print(f"{'#'*80}")
            try:
                result = run_simulation_for_frequency(
                    freq, num_cycles=num_cycles, particle_density=2000, num_particles=N_PARTICLES
                )
                result_list.append(result)
                if result is not None:
                    print(f"\n✓ frequency {freq} Hz ({num_cycles} cycles) done")
                else:
                    print(f"\n⚠ frequency {freq} Hz failed")
            except KeyboardInterrupt:
                print(f"\n\nInterrupted after {i-1}/{len(FREQUENCIES)} frequencies")
                while len(result_list) < len(FREQUENCIES):
                    result_list.append(None)
                break
            except Exception as e:
                print(f"\n✗ frequency {freq} Hz : {e}")
                import traceback
                traceback.print_exc()
                result_list.append(None)
            done_wall = time.perf_counter() - wall_start
            print(
                f"Overall progress: {case_i}/{case_total} cases | "
                f"elapsed {done_wall/60:.1f} min"
            )

    #
    while len(all_results_100) < len(FREQUENCIES):
        all_results_100.append(None)
    while len(all_results_1000) < len(FREQUENCIES):
        all_results_1000.append(None)

    wall_elapsed = time.perf_counter() - wall_start

    # Save results
    has_any = any(r is not None for r in all_results_100) or any(r is not None for r in all_results_1000)
    if has_any:
        print(f"\n\n{'='*80}")
        print("...")
        print(f"{'='*80}")
        
        output_dir = Path(__file__).parent
        save_results(all_results_100, all_results_1000, output_dir)
        
        print(f"\n\n{'='*80}")
        print("frequency！")
        print(f"{'='*80}")
        
        #
        valid_1000 = [r for r in all_results_1000 if r is not None]
        valid_100 = [r for r in all_results_100 if r is not None]
        valid_results = valid_1000 or valid_100
        num_particles_used = valid_results[0]['num_particles'] if valid_results else N_PARTICLES
        total_comp_time = sum(r['computation_time_s'] for r in valid_100) + sum(r['computation_time_s'] for r in valid_1000)
        
        print("\n" + "="*60)
        print("")
        print("="*60)
        print(f"   (num_particles):     {num_particles_used}")
        print(f"   (num_cycles):      100  1000()")
        print(f"  frequency:                    {len(FREQUENCIES)}")
        print(f"  frequency (1000 ):   {len(valid_1000)}")
        print(f"  frequency (100 ):   {len(valid_100)}")
        print(f"   SPL (dB):                 {SPL_DB}")
        print(f"   (frequency): {total_comp_time:.2f} s")
        print(f"   ():      {wall_elapsed:.2f} s ({wall_elapsed/60:.2f} min)")
        print("="*60)
        print("\nfrequency (1000 ):")
        for result in valid_1000:
            print(f"  {result['frequency_hz']:5d} Hz ({result['frequency_khz']}k): "
                  f" {result['computation_time_s']:.2f} s, "
                  f" {result['total_simulation_time']:.6f} s, "
                  f"Peak={result['peak_increase_percent']:.2f}%, CV={result['concentration_cv']:.4f}")
        for i, r in enumerate(all_results_1000):
            if r is None:
                print(f"  {FREQUENCIES[i]} Hz: ")
        print("="*60)
        
        print("\n (1000 ):")
        for result in all_results_1000:
            if result is not None:
                freq = result['frequency_hz']
                peak = result['peak_increase_percent']
                cv = result['concentration_cv']
                print(f"  {freq:5d} Hz ({freq//1000:2d}k): Peak={peak:6.2f}%, CV={cv:.4f}")
            else:
                print(f"  frequency")
        
        print(f"\n:")
        print(f"  - {output_dir / 'aggregation_metrics_100cycles.csv'}")
        print(f"  - {output_dir / 'aggregation_metrics_1000cycles.csv'}")
        print(f"  - {output_dir / 'concentration_distribution_combined_100cycles.png'} (100 )")
        print(f"  - {output_dir / 'concentration_distribution_combined_1000cycles.png'} (1000 )")
        print(f"  - {output_dir / 'aggregation_metrics_plot.png'} (:)")
    else:
        print("\n✗ frequency,")


if __name__ == "__main__":
    main()
