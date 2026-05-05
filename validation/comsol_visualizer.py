"""Visualize COMSOL-exported particle positions along x.

Reads a CSV where column 0 is time [s] and remaining columns are particle x [mm]
in [0, 34]. Produces:
  - validation/comsol_density_evolution.png — absolute density vs x
  - validation/comsol_density_change.png — percent change vs initial time

Example CSV layout (header optional; script uses header=None):
    time_s, p1_mm, p2_mm, ...
    0.000, 5.2, 8.1, ...

Run from repository root:
    python validation/comsol_visualizer.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_particle_density(particle_positions_mm, x_grid_mm, bandwidth=0.5):
    """
    Gaussian kernel estimate of number density along x.

    Kernel: exp(-(x - x_i)^2 / (2 * bandwidth^2)), summed over particles.

    :param particle_positions_mm: 1D array of x positions [mm]
    :param x_grid_mm: evaluation grid [mm]
    :param bandwidth: Gaussian sigma [mm]
    :return: density at each grid point (unnormalized count-like units)
    """
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)

    particle_positions_mm = particle_positions_mm[~np.isnan(particle_positions_mm)]
    particle_positions_mm = particle_positions_mm[
        (particle_positions_mm >= 0) & (particle_positions_mm <= 34)
    ]

    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)

    density = np.zeros_like(x_grid_mm)
    for idx, x_pos in enumerate(x_grid_mm):
        distances = np.abs(particle_positions_mm - x_pos)
        density[idx] = np.sum(np.exp(-distances ** 2 / (2 * bandwidth ** 2)))

    return density


def main():
    print("=" * 80)
    print("COMSOL particle positions — density visualization")
    print("=" * 80)

    csv_path = "validation/comsol_positions.csv"

    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}")
        return

    print(f"\nReading: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    num_timesteps = len(df)
    num_particles = len(df.columns) - 1

    print(f"Shape: {df.shape}")
    print(f"Detected {num_timesteps} time rows")
    print(f"Detected {num_particles} particle columns")

    if num_timesteps == 0:
        print("Error: no data rows in CSV")
        return
    if num_particles == 0:
        print("Error: no particle columns after time column")
        return

    time_values = df.iloc[:, 0].values

    nan_count = int(np.sum(np.isnan(time_values)))
    if nan_count > 0:
        print(f"\nWarning: {nan_count} NaN entries in time column")
        valid_indices = ~np.isnan(time_values)
        time_values = time_values[valid_indices]
        particle_positions = df.iloc[:, 1:].values[valid_indices, :]
        num_timesteps = len(time_values)
        print(f"Continuing with {num_timesteps} valid time rows")
        if num_timesteps == 0:
            print("Error: no valid times after filtering")
            return
    else:
        particle_positions = df.iloc[:, 1:].values

    print(f"\nTime span: {time_values[0]:.6f} s — {time_values[-1]:.6f} s")

    if num_timesteps <= 20:
        print(f"Times: {time_values}")
    else:
        print(f"First 5 times: {time_values[:5]}")
        print(f"Last 5 times: {time_values[-5:]}")

    x_grid_mm = np.linspace(0, 34, 200)

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_timesteps))

    print(f"\n{'=' * 80}")
    print(f"Computing density at {num_timesteps} times...")
    print(f"{'=' * 80}\n")

    all_densities = []

    for i, t in enumerate(time_values):
        positions_at_t = particle_positions[i, :]
        density = calculate_particle_density(positions_at_t, x_grid_mm, bandwidth=0.5)
        all_densities.append(density)
        density_smoothed = gaussian_filter1d(density, sigma=2)
        label = f"t = {t:.4f} s ({t * 1000:.2f} ms)"
        ax.plot(
            x_grid_mm,
            density_smoothed,
            color=colors[i],
            linewidth=2.0,
            alpha=0.8,
            label=label,
        )
        print(
            f"Step {i + 1}/{len(time_values)}: t={t:.6f} s, "
            f"valid particles={np.sum(~np.isnan(positions_at_t))}, "
            f"max density={np.max(density):.2f}"
        )

    ax.set_xlim(0, 34)
    ax.set_xlabel("Position x (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Particle number density (a.u.)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"COMSOL export: density evolution ({num_timesteps} times)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    ncol = 2 if num_timesteps <= 12 else 3 if num_timesteps <= 24 else 4
    ax.legend(loc="best", fontsize=9, framealpha=0.9, ncol=ncol)

    all_densities_array = np.array(all_densities)
    y_min = float(np.min(all_densities_array))
    y_max = float(np.max(all_densities_array))
    y_range = y_max - y_min
    y_margin = max(y_range * 0.1, 1.0)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()

    output_dir = "validation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comsol_density_evolution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"\n{'=' * 80}")
    print(f"Figure 1 saved: {output_path}")
    print(f"{'=' * 80}\n")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    initial_density = all_densities[0]

    print(f"{'=' * 80}")
    print(f"Relative change vs t = {time_values[0]:.6f} s")
    print(f"{'=' * 80}\n")

    for i, t in enumerate(time_values):
        density = all_densities[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            relative_change = (density - initial_density) / (initial_density + 1e-10) * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)

        label = f"t = {t:.4f} s ({t * 1000:.2f} ms)"
        ax2.plot(
            x_grid_mm,
            relative_change,
            color=colors[i],
            linewidth=2.0,
            alpha=0.8,
            label=label,
        )
        print(
            f"Step {i + 1}/{num_timesteps}: t={t:.6f} s, "
            f"relative change [%]: [{np.min(relative_change):.2f}, {np.max(relative_change):.2f}]"
        )

    ax2.set_xlim(0, 34)
    ax2.set_xlabel("Position x (mm)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Relative density change (%)", fontsize=14, fontweight="bold")
    ax2.set_title(
        f"COMSOL export: density change vs initial time ({num_timesteps} curves)",
        fontsize=16,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3, linestyle="--")

    ncol2 = 2 if num_timesteps <= 12 else 3 if num_timesteps <= 24 else 4
    ax2.legend(loc="best", fontsize=9, framealpha=0.9, ncol=ncol2)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    output_path2 = os.path.join(output_dir, "comsol_density_change.png")
    plt.savefig(output_path2, dpi=300, bbox_inches="tight")

    print(f"\nFigure 2 saved: {output_path2}")
    print(f"\n{'=' * 80}")
    print("Done.")
    print(f"  Time rows: {num_timesteps}")
    print(f"  Particles: {num_particles}")
    print(f"{'=' * 80}\n")
    plt.close()


if __name__ == "__main__":
    main()
