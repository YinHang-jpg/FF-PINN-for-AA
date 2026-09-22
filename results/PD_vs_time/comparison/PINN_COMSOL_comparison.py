from pathlib import Path
import sys
_RESULTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RESULTS))
from _figures import FIGURES
"""
PINN vs FEM (finite-element reference) particle-position comparison

This script compares particle number concentration distributions and computational time between
PINN, DEM, and a FEM reference (e.g. COMSOL-exported trajectories saved as CSV).

FEM trajectories must live under ``results/PD_vs_time/`` in the same row layout as
``pinn_particle_positions.csv`` (time in column 0, particle x in mm in columns 1..N,
0–34 mm domain). Preferred filename: ``fem_particle_positions.csv``.

Time is distinguished by marker shape (and color on the error plot), not by dashed line styles,
so legends remain readable when printed or viewed at small size.

Features:
    1. Plot 1: Concentration comparison at t=0.01, 0.03, 0.05s (6 curves total)
    2. Plot 2: Concentration comparison at t=0.06, 0.08, 0.1s (6 curves total)
    3. Plot 3: Error analysis between PINN and FEM reference
    4. Plot 4: Computational time comparison (linear fit)

Output (under figures/):
    - fig08_density_comparison_group1.png
    - fig08_density_comparison_group2.png
    - fig08_density_error_analysis.png
    - fig12_computation_time_comparison.png  (manuscript Fig. 12)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d

# Time is encoded with markers (legend + plot are easier to read than dash / dash-dot alone).
TIME_MARKERS_3 = ["o", "s", "^"]  # circle, square, triangle_up
TIME_MARKERS_6 = ["o", "s", "^", "D", "v", "P"]


def resolve_fem_positions_csv(parent_dir):
    """
    Return path to the FEM reference particle CSV (same format as pinn_particle_positions.csv).

    Resolution order:
        1. results/PD_vs_time/fem_particle_positions.csv
        2. results/PD_vs_time/comsol_particle_positions.csv
        3. Path in environment variable FEM_PARTICLE_POSITIONS_CSV (if set and existing)

    ``validation/comsol_positions.csv`` may be copied to ``fem_particle_positions.csv``
    when the export already uses the 0–34 mm channel geometry.
    """
    candidates = [
        os.path.join(parent_dir, "fem_particle_positions.csv"),
        os.path.join(parent_dir, "comsol_particle_positions.csv"),
    ]
    env_path = os.environ.get("FEM_PARTICLE_POSITIONS_CSV", "").strip()
    if env_path:
        candidates.append(env_path)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def calculate_particle_density(particle_positions_mm, x_grid_mm, sigma=0.3):
    """
    Calculate particle number density using Gaussian kernel density estimation
    
    Parameters:
        particle_positions_mm: 1D array containing x-coordinates of all particles (unit: mm)
        x_grid_mm: 1D array of x-axis grid points for density calculation (unit: mm)
        sigma: Standard deviation of Gaussian kernel (unit: mm)
    
    Returns:
        density: Particle number density at each point in x_grid_mm
    """
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    # Remove invalid values
    particle_positions_mm = particle_positions_mm[~np.isnan(particle_positions_mm)]
    particle_positions_mm = particle_positions_mm[(particle_positions_mm >= 0) & (particle_positions_mm <= 34)]
    
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    density = np.zeros_like(x_grid_mm)
    
    # Gaussian kernel density estimation
    for idx, x_pos in enumerate(x_grid_mm):
        distances = np.abs(particle_positions_mm - x_pos)
        density_contribution = np.exp(-distances**2 / (2 * sigma**2))
        density[idx] = np.sum(density_contribution)
    
    return density


def find_closest_time_index(target_time, time_array):
    """Find the index of the closest time value in the array"""
    return np.argmin(np.abs(time_array - target_time))


def load_csv_data(csv_path):
    """Load CSV data and return time values and particle positions"""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    
    time_values = df.iloc[:, 0].values
    particle_positions = df.iloc[:, 1:].values
    
    # Filter NaN timesteps
    valid_indices = ~np.isnan(time_values)
    time_values = time_values[valid_indices]
    particle_positions = particle_positions[valid_indices, :]
    
    print(f"  ✓ Loaded {len(time_values)} timesteps, {particle_positions.shape[1]} particles")
    print(f"  Time range: {time_values[0]:.6f}s to {time_values[-1]:.6f}s")
    
    return time_values, particle_positions


def plot_density_comparison(pinn_data, comsol_data, dem_csv_path, target_times, output_filename, group_name):
    """
    Plot concentration comparison between PINN, DEM, and FEM reference for specified times
    
    Parameters:
        pinn_data: tuple of (time_values, particle_positions)
        comsol_data: tuple of (time_values, particle_positions)
        dem_csv_path: path to DEM particle distribution CSV file
        target_times: list of target times to plot
        output_filename: output file path
        group_name: name of the group for title
    """
    pinn_time, pinn_positions = pinn_data
    comsol_time, comsol_positions = comsol_data
    
    # Load DEM data from CSV
    dem_df = pd.read_csv(dem_csv_path)
    x_dem_mm = dem_df['x_mm'].values
    
    # Map target times to column names for DEM data
    # target_times: [0.01, 0.03, 0.05] -> steps: [1e4, 3e4, 5e4] or [0.06, 0.08, 0.1] -> steps: [6e4, 8e4, 1e5]
    time_to_column = {
        0.01: 'relative_change_1.0e+04',
        0.03: 'relative_change_3.0e+04',
        0.05: 'relative_change_5.0e+04',
        0.06: 'relative_change_6.0e+04',
        0.08: 'relative_change_8.0e+04',
        0.1: 'relative_change_1.0e+05'
    }
    
    # Define x-axis grid
    x_grid_mm = np.linspace(0, 34, 200)
    x_mask = (x_grid_mm >= 2) & (x_grid_mm <= 32)
    x_grid_cropped = x_grid_mm[x_mask]
    
    # Calculate initial densities
    pinn_initial_density = calculate_particle_density(pinn_positions[0, :], x_grid_mm, sigma=0.3)
    comsol_initial_density = calculate_particle_density(comsol_positions[0, :], x_grid_mm, sigma=0.3)
    
    n_x = len(x_grid_cropped)
    markevery = max(1, n_x // 10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    print(f"\nGenerating {group_name}...")
    
    # Store handles and labels for unified legend
    all_handles = []
    all_labels = []
    
    for idx, target_t in enumerate(target_times):
        # PINN data
        pinn_idx = find_closest_time_index(target_t, pinn_time)
        pinn_actual_t = pinn_time[pinn_idx]
        pinn_pos = pinn_positions[pinn_idx, :]
        pinn_density = calculate_particle_density(pinn_pos, x_grid_mm, sigma=0.3)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pinn_relative_change = (pinn_density - pinn_initial_density) / pinn_initial_density * 100.0
            pinn_relative_change = np.nan_to_num(pinn_relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        pinn_relative_cropped = pinn_relative_change[x_mask]
        
        # COMSOL data
        comsol_idx = find_closest_time_index(target_t, comsol_time)
        comsol_actual_t = comsol_time[comsol_idx]
        comsol_pos = comsol_positions[comsol_idx, :]
        comsol_density = calculate_particle_density(comsol_pos, x_grid_mm, sigma=0.3)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            comsol_relative_change = (comsol_density - comsol_initial_density) / comsol_initial_density * 100.0
            comsol_relative_change = np.nan_to_num(comsol_relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        comsol_relative_cropped = comsol_relative_change[x_mask]
        
        # DEM data - load from CSV and flip horizontally
        dem_column = time_to_column[target_t]
        dem_relative_change = dem_df[dem_column].values
        
        # Flip DEM data horizontally (mirror)
        # Original: x goes from 2mm to 32mm
        # Flipped: flip around center (17mm)
        center_x = 17.0  # (2 + 32) / 2
        x_dem_flipped = 2 * center_x - x_dem_mm  # Mirror around center
        
        # Interpolate to match the cropped grid
        dem_interp = interp1d(x_dem_flipped, dem_relative_change, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        dem_relative_cropped = dem_interp(x_grid_cropped)
        
        m = TIME_MARKERS_3[idx % len(TIME_MARKERS_3)]
        mk = dict(
            markevery=markevery,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        
        # Plot PINN (red, thickest): color = method, marker = time
        line_pinn, = ax.plot(
            x_grid_cropped,
            pinn_relative_cropped,
            color="red",
            linestyle="-",
            linewidth=3.2,
            alpha=0.9,
            marker=m,
            markeredgecolor="red",
            **mk,
            label=f"PINN t={pinn_actual_t:.2f}s",
        )
        
        # Plot DEM (green, medium)
        line_dem, = ax.plot(
            x_grid_cropped,
            dem_relative_cropped,
            color="green",
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
            marker=m,
            markeredgecolor="green",
            **mk,
            label=f"DEM t={target_t:.2f}s",
        )
        
        # Plot FEM reference (blue, thinnest)
        line_comsol, = ax.plot(
            x_grid_cropped,
            comsol_relative_cropped,
            color="blue",
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
            marker=m,
            markeredgecolor="blue",
            **mk,
            label=f"FEM t={comsol_actual_t:.2f}s",
        )
        
        # Add to unified legend in order: PINN, DEM, FEM for each time
        all_handles.extend([line_pinn, line_dem, line_comsol])
        all_labels.extend([f'PINN t={pinn_actual_t:.2f}s', 
                          f'DEM t={target_t:.2f}s',
                          f'FEM t={comsol_actual_t:.2f}s'])
        
        print(f"  t={target_t:.2f}s: PINN range=[{np.min(pinn_relative_cropped):.2f}%, {np.max(pinn_relative_cropped):.2f}%], "
              f"DEM range=[{np.min(dem_relative_cropped):.2f}%, {np.max(dem_relative_cropped):.2f}%], "
              f"FEM range=[{np.min(comsol_relative_cropped):.2f}%, {np.max(comsol_relative_cropped):.2f}%]")
    
    # Set plot properties
    ax.set_xlim(2, 32)
    ax.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Concentration Change (%)', fontsize=14, fontweight='bold')
    ax.set_title(group_name, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create unified legend at upper center (handlelength shows markers clearly)
    leg = ax.legend(
        all_handles,
        all_labels,
        loc="upper center",
        fontsize=10,
        framealpha=0.95,
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        handlelength=3.2,
        handletextpad=0.6,
        labelspacing=0.8,
    )
    
    
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_filename}")
    plt.close()


def plot_error_analysis(pinn_data, comsol_data, target_times, output_filename,
                        sigma_kde_mm=0.6):
    """
    Pointwise spatial error between PINN and FEM particle number-density profiles.

    Metric (signed, percentage; axis label: "Normalized Error (%)"):

        rho_pinn_hat(x,t) = rho_pinn(x,t) / sum_window( rho_pinn(x,t) )       # unit area
        rho_fem_hat(x,t)  = rho_fem(x,t)  / sum_window( rho_fem(x,t)  )
        rho0_hat          = window-mean of  ( rho_pinn_hat(x,0) + rho_fem_hat(x,0) ) / 2
        error(x,t)        = ( rho_pinn_hat(x,t) - rho_fem_hat(x,t) ) / rho0_hat * 100 %

    Densities are area-normalized inside [2, 32] mm so the comparison reflects
    profile shape. The denominator maps the error onto the same
    ``(rho - rho_0) / rho_0`` percentage scale used in panels (a) and (b).
    """
    pinn_time, pinn_positions = pinn_data
    comsol_time, comsol_positions = comsol_data

    x_grid_mm = np.linspace(0, 34, 400)
    x_mask = (x_grid_mm >= 2) & (x_grid_mm <= 32)
    x_grid_cropped = x_grid_mm[x_mask]

    err_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    err_styles = ["-", "--", "-.", ":", "-", "--"]

    fig, ax = plt.subplots(figsize=(12, 7))
    print("\nGenerating error analysis...")

    pinn_initial_density = calculate_particle_density(pinn_positions[0, :], x_grid_mm, sigma=sigma_kde_mm)
    fem_initial_density = calculate_particle_density(comsol_positions[0, :], x_grid_mm, sigma=sigma_kde_mm)

    pinn_init_norm = max(float(pinn_initial_density[x_mask].sum()), 1e-12)
    fem_init_norm = max(float(fem_initial_density[x_mask].sum()), 1e-12)
    pinn_init_hat = pinn_initial_density / pinn_init_norm
    fem_init_hat = fem_initial_density / fem_init_norm

    rho0_hat = float(np.mean(0.5 * (pinn_init_hat[x_mask] + fem_init_hat[x_mask])))
    if rho0_hat <= 0:
        rho0_hat = 1e-12
    print(f"Reference initial density (window mean of area-normalized profiles): {rho0_hat:.6e}")

    for idx, target_t in enumerate(target_times):
        pinn_idx = find_closest_time_index(target_t, pinn_time)
        pinn_density = calculate_particle_density(pinn_positions[pinn_idx, :], x_grid_mm, sigma=sigma_kde_mm)
        pinn_hat = pinn_density / max(float(pinn_density[x_mask].sum()), 1e-12)

        fem_idx = find_closest_time_index(target_t, comsol_time)
        fem_density = calculate_particle_density(comsol_positions[fem_idx, :], x_grid_mm, sigma=sigma_kde_mm)
        fem_hat = fem_density / max(float(fem_density[x_mask].sum()), 1e-12)

        err = (pinn_hat - fem_hat) / rho0_hat * 100.0
        err_cropped = err[x_mask]

        c = err_colors[idx % len(err_colors)]
        ls = err_styles[idx % len(err_styles)]
        ax.plot(
            x_grid_cropped,
            err_cropped,
            color=c,
            linestyle=ls,
            linewidth=2.0,
            alpha=0.95,
            label=f"t={target_t:.2f}s",
        )

        rmse = float(np.sqrt(np.mean(err_cropped**2)))
        mae = float(np.mean(np.abs(err_cropped)))
        max_err = float(np.max(np.abs(err_cropped)))
        print(f"  t={target_t:.2f}s: RMSE={rmse:.2f}%, MAE={mae:.2f}%, Max|err|={max_err:.2f}%")

    ax.set_xlim(2, 32)
    ax.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Error (%)', fontsize=14, fontweight='bold')
    ax.set_title('(c)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9, handlelength=3.0, handletextpad=0.55)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_filename}")
    plt.close()


def plot_computation_time(output_filename):
    """Plot Fig. 12 from archived wall-clock CSV tables (see other_files/)."""
    from pathlib import Path as _Path
    import importlib.util

    script = _Path(__file__).resolve().parents[3] / "other_files" / "plot_time_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_time_comparison", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    st, pinn, dem, comsol = mod.read_simtime_csv(mod.SIMTIME_CSV)
    mod.plot_simtime(st, pinn, dem, comsol, _Path(output_filename))

    particle_out = _Path(output_filename).with_name(
        "fig12_computation_time_vs_particle_count.png"
    )
    dx, dy, px, py, cx, cy = mod.read_particle_csv(mod.PARTICLE_CSV)
    mod.plot_particle(dx, dy, px, py, cx, cy, particle_out)
    print(f"✓ Saved: {output_filename}")
    print(f"✓ Saved: {particle_out}")


def main():
    """Main function"""
    print("="*80)
    print("PINN vs FEM Comparison Analysis")
    print("="*80)
    
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # PD_vs_time folder
    pinn_csv = os.path.join(parent_dir, 'pinn_particle_positions.csv')
    dem_csv = os.path.join(parent_dir, 'dem_particle_distribution_data.csv')
    comsol_csv = resolve_fem_positions_csv(parent_dir)
    
    # Check files exist
    if not os.path.exists(pinn_csv):
        print(f"Error: File not found {pinn_csv}")
        return
    if not os.path.exists(dem_csv):
        print(f"Error: File not found {dem_csv}")
        return
    if not comsol_csv:
        print(
            "Error: FEM particle CSV not found. Expected "
            "results/PD_vs_time/fem_particle_positions.csv "
            "(same layout as pinn_particle_positions.csv)."
        )
        return
    
    # Load data
    print("\n" + "="*80)
    print("Loading CSV files...")
    print("="*80)
    pinn_time, pinn_positions = load_csv_data(pinn_csv)
    comsol_time, comsol_positions = load_csv_data(comsol_csv)
    
    # Set matplotlib parameters
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Define target times
    target_times_group1 = [0.01, 0.03, 0.05]
    target_times_group2 = [0.06, 0.08, 0.1]
    all_target_times = target_times_group1 + target_times_group2
    
    # Plot 1: Group 1 comparison
    print("\n" + "="*80)
    plot_density_comparison(
        (pinn_time, pinn_positions),
        (comsol_time, comsol_positions),
        dem_csv,
        target_times_group1,
        str(FIGURES / 'fig08_density_comparison_group1.png'),
        '(a)'
    )
    
    # Plot 2: Group 2 comparison
    plot_density_comparison(
        (pinn_time, pinn_positions),
        (comsol_time, comsol_positions),
        dem_csv,
        target_times_group2,
        str(FIGURES / 'fig08_density_comparison_group2.png'),
        '(b)'
    )
    
    # Plot 3: Error analysis
    print("="*80)
    plot_error_analysis(
        (pinn_time, pinn_positions),
        (comsol_time, comsol_positions),
        all_target_times,
        str(FIGURES / 'fig08_density_error_analysis.png')
    )
    
    # Plot 4: Computation time comparison (manuscript Fig. 12)
    print("="*80)
    plot_computation_time(
        str(FIGURES / 'fig12_computation_time_comparison.png')
    )
    
    print("\n" + "="*80)
    print("✓ All plots generated successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
