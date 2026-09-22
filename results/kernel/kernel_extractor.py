"""
Agglomeration kernel estimator driven by PINN models.

This script loads the frequency-specific PINN models, samples particles of
various diameters, runs short deterministic integrations using the unified
physical model, and converts the resulting velocity statistics into an
effective agglomeration kernel K(d1, d2).

Outputs:
    - Heat map of K(d1,d2) for each processed frequency
    - Reference kernel (simplified Hoffmann-like form) for comparison
    - Average collision rate versus time curves
    - NumPy archive containing all intermediate data
"""

from __future__ import annotations

import os
import sys
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


# ------------------------------------------------------------------------------
# Project paths and imports
# ------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from initialization.particle_initialization import initialize_particles
from mechanisms.UNIFIED_PINN import (
    load_individual_models,
    UnifiedNormalizer,
    PhysicalUnifiedModel,
)


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Excitation frequencies for orthokinetic kernel study (paper: 8–16 kHz).
FREQUENCIES = [8000, 10000, 12000, 16000]
FREQUENCY_FILES = [
    "initialization/sound_source_standing.py",
    "mechanisms/ARF.py",
    "mechanisms/STOKES_PINN_t.py",
    "mechanisms/STOKES_PINN_v.py",
    "mechanisms/STOKES_PINN_x.py",
]

REQUIRED_MODEL_FILES = [
    "arf_model_x.pth",
    "arf_model_x_normalization_params.json",
    "arf_model_t.pth",
    "arf_model_t_normalization_params.json",
    "stokes_model_x.pth",
    "stokes_model_x_normalization_params.json",
    "stokes_model_t.pth",
    "stokes_model_t_normalization_params.json",
    "stokes_model_v.pth",
    "stokes_model_v_normalization_params.json",
]

DIAMETER_GRID_UM = np.array([0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
SAMPLES_PER_DIAMETER = 1024
SIMULATION_STEPS = 10000  # Reduced from 50000 to 10000 for faster computation
RECORD_INTERVAL = 2       # Very small interval (2 µs) for nearly continuous data
DT = 1.0e-6
DOMAIN_SIZE = (0.034, 0.034)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collision adhesion probability P_coll(|Δv|) = 1 - exp(-|Δv| / v_s) with v_s = 0.02 m/s (manuscript).
RELATIVE_SPEED_SCALE = 0.02  # m/s, v_s
REF_DIAMETER_M = 2.0e-6  # m

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed color scale for `kernel_heatmap_freq_*k.png` across all frequencies.
# Plotted values are K and (K - K_ref) in units of 10^-12 m^3/s (same as colorbar labels).
HEATMAP_K_VMIN = 0.0
HEATMAP_K_VMAX = 80.0
# Diverging panel: symmetric range [-HEATMAP_DELTA_VMAX, +HEATMAP_DELTA_VMAX].
HEATMAP_DELTA_VMAX = 80.0


# ------------------------------------------------------------------------------
# Utility functions: frequency configuration
# ------------------------------------------------------------------------------

def update_frequency_in_file(file_path: str, new_frequency: int) -> bool:
    """Update the `frequency` constant in the given file (silently)."""
    full_path = PROJECT_ROOT / file_path
    if not full_path.exists():
        return False

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"frequency\s*=\s*\d+"
    replacement = f"frequency = {new_frequency}"
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def backup_frequency_files() -> Dict[str, Path]:
    """Backup files that need in-place frequency updates."""
    backups: Dict[str, Path] = {}
    for rel_path in FREQUENCY_FILES:
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            continue
        backup = PROJECT_ROOT / f"{rel_path}.kernelbak"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, backup)
        backups[rel_path] = backup
    return backups


def restore_frequency_files(backups: Dict[str, Path]) -> None:
    """Restore previously backed-up frequency configuration files."""
    for rel_path, backup in backups.items():
        src = PROJECT_ROOT / rel_path
        if backup.exists():
            shutil.copy2(backup, src)
            backup.unlink()


def ensure_models_for_frequency(frequency: int) -> bool:
    """Copy trained models for a specific frequency into the active PINN directory (silently)."""
    freq_k = frequency // 1000
    freq_dir = PROJECT_ROOT / "PINN" / f"freq_{freq_k}k"
    pinn_dir = PROJECT_ROOT / "PINN"

    if not freq_dir.exists():
        return False

    for fname in REQUIRED_MODEL_FILES:
        src = freq_dir / fname
        dst = pinn_dir / fname
        if not src.exists():
            return False
        shutil.copy2(src, dst)

    return True


# ------------------------------------------------------------------------------
# PINN model helpers
# ------------------------------------------------------------------------------

def build_physical_model(device: torch.device) -> PhysicalUnifiedModel:
    """Instantiate the physical unified PINN model with current frequency files."""
    models = load_individual_models(device=device, model_base_path=str(PROJECT_ROOT / "PINN"))
    unified_norm = UnifiedNormalizer()
    model_paths = {
        "arf_x_norm": PROJECT_ROOT / "PINN" / "arf_model_x_normalization_params.json",
        "arf_t_norm": PROJECT_ROOT / "PINN" / "arf_model_t_normalization_params.json",
        "stokes_x_norm": PROJECT_ROOT / "PINN" / "stokes_model_x_normalization_params.json",
        "stokes_t_norm": PROJECT_ROOT / "PINN" / "stokes_model_t_normalization_params.json",
        "stokes_v_norm": PROJECT_ROOT / "PINN" / "stokes_model_v_normalization_params.json",
    }
    model_paths = {k: str(v) for k, v in model_paths.items()}
    unified_norm.load_from_individual_models(model_paths, device="cpu")
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
    phys_model.eval()
    return phys_model


# ------------------------------------------------------------------------------
# Simulation utilities
# ------------------------------------------------------------------------------

def simulate_velocity_history(
    diameter_m: float,
    phys_model: PhysicalUnifiedModel,
    device: torch.device,
    steps: int,
    record_interval: int,
    samples: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run a short deterministic integration to capture velocity statistics."""
    positions_np, velocities_np, radii_np, mass_np = initialize_particles(
        N=samples,
        domain_size=DOMAIN_SIZE,
        diameter=diameter_m,
        init_mode="random",
    )

    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)
    inv_mass = 1.0 / mass_t

    velocity_history: List[np.ndarray] = []
    time_points: List[float] = []

    with torch.inference_mode():
        for step in range(steps):
            current_time = step * DT
            fx = phys_model(
                positions_t[:, 0],
                velocities_t[:, 0],
                torch.full_like(positions_t[:, 0], current_time),
            )

            velocities_t[:, 0].add_(fx * inv_mass, alpha=DT)
            positions_t.add_(velocities_t, alpha=DT)

            if step % record_interval == 0:
                velocity_history.append(velocities_t.detach().cpu().numpy().copy())
                time_points.append(current_time)

    return np.array(time_points), velocity_history


def estimate_relative_speed(
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    max_pairs: int = 1024,  # Increased from 256 to 1024 for better statistical sampling
) -> float:
    """Estimate mean relative speed between two velocity samples."""
    if vel_a.size == 0 or vel_b.size == 0:
        return 0.0

    count = min(len(vel_a), len(vel_b), max_pairs)
    idx_a = np.random.choice(len(vel_a), count, replace=False)
    idx_b = np.random.choice(len(vel_b), count, replace=False)
    rel = vel_a[idx_a] - vel_b[idx_b]
    rel_speed = np.linalg.norm(rel, axis=1)
    return float(np.maximum(np.mean(rel_speed), 1e-6))


def diameter_pair_enhancement(diameter_sum_m: float) -> float:
    """Identity factor (kept for call-site compatibility)."""
    return 1.0


def collision_probability(rel_speed: float) -> float:
    """Empirical adhesion probability P_coll = 1 - exp(-|Δv|/v_s), manuscript Sec. 3.3."""
    vs = max(RELATIVE_SPEED_SCALE, 1e-12)
    prob = 1.0 - float(np.exp(-rel_speed / vs))
    return float(np.clip(prob, 0.0, 1.0))


def hoffmann_reference_kernel(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """
    Simplified Hoffmann-style reference kernel.

    Reference form: K = π (r1 + r2)^2 * (u_acoustic + u_brownian)
    with heuristic acoustic drift velocity.
    """
    r_sum = (d1 + d2) / 2.0
    cross_section = np.pi * (r_sum / 2.0) ** 2
    acoustic_velocity = 0.015 * ((d1 + d2) / (2 * REF_DIAMETER_M)) ** 0.3
    brownian_velocity = 0.005 / np.sqrt((d1 * d2) / (REF_DIAMETER_M**2))
    total_velocity = acoustic_velocity + brownian_velocity
    return cross_section * total_velocity


# ------------------------------------------------------------------------------
# Kernel computation & plotting
# ------------------------------------------------------------------------------

def compute_kernel_matrices(
    diameters_um: np.ndarray,
    time_points: np.ndarray,
    velocity_history: Dict[int, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-resolved kernel matrices."""
    num_records = len(time_points)
    num_d = len(diameters_um)
    kernels = np.zeros((num_records, num_d, num_d), dtype=np.float64)

    diameters_m = diameters_um * 1e-6

    for record_idx in range(num_records):
        for i in range(num_d):
            vel_i = velocity_history[i][record_idx]
            for j in range(num_d):
                vel_j = velocity_history[j][record_idx]
                rel_speed = estimate_relative_speed(vel_i, vel_j)
                cross_section = np.pi * ((diameters_m[i] + diameters_m[j]) / 2.0) ** 2
                p_coll = collision_probability(rel_speed)
                d_enh = diameter_pair_enhancement(diameters_m[i] + diameters_m[j])
                kernels[record_idx, i, j] = cross_section * rel_speed * p_coll * d_enh

    theory_kernel = hoffmann_reference_kernel(
        diameters_m[:, None], diameters_m[None, :]
    )
    return kernels, theory_kernel


def plot_kernel_heatmap(
    diameters_um: np.ndarray,
    kernel_matrix: np.ndarray,
    theory_matrix: np.ndarray,
    frequency: int,
) -> None:
    """Save a comparison heatmap between PINN-derived and reference kernels (silently)."""
    freq_k = frequency // 1000
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = [diameters_um[0], diameters_um[-1], diameters_um[0], diameters_um[-1]]

    im0 = axes[0].imshow(
        kernel_matrix * 1e12,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="inferno",
        vmin=HEATMAP_K_VMIN,
        vmax=HEATMAP_K_VMAX,
    )
    axes[0].set_title(f"PINN Kernel K(d1,d2) @ {freq_k} kHz")
    axes[0].set_xlabel("d1 (μm)")
    axes[0].set_ylabel("d2 (μm)")
    fig.colorbar(im0, ax=axes[0], label=r"K [$10^{-12}$ m$^3$/s]")

    delta = (kernel_matrix - theory_matrix) * 1e12
    im1 = axes[1].imshow(
        delta,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-HEATMAP_DELTA_VMAX,
        vmax=HEATMAP_DELTA_VMAX,
    )
    axes[1].set_title("PINN - Reference (×1e-12 m³/s)")
    axes[1].set_xlabel("d1 (μm)")
    axes[1].set_ylabel("d2 (μm)")
    fig.colorbar(
        im1,
        ax=axes[1],
        label=r"$\Delta$K [$10^{-12}$ m$^3$/s]",
    )

    out_path = OUTPUT_DIR / f"kernel_heatmap_freq_{freq_k}k.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Main processing
# ------------------------------------------------------------------------------


def process_frequency(frequency: int, freq_idx: int, total_freqs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single frequency and return time_points and collision rates for plotting."""
    freq_k = frequency // 1000
    
    # Progress bar
    progress = (freq_idx + 1) / total_freqs * 100
    bar_length = 40
    filled = int(bar_length * (freq_idx + 1) / total_freqs)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r[{bar}] {progress:.0f}% | Processing {freq_k} kHz", end='', flush=True)

    if not ensure_models_for_frequency(frequency):
        print(f"\n✗ Error: Models not available for {freq_k} kHz")
        return None, None

    # Update frequency configuration (silently)
    for file_path in FREQUENCY_FILES:
        update_frequency_in_file(file_path, frequency)

    # Build model
    phys_model = build_physical_model(DEVICE)

    velocity_history: Dict[int, List[np.ndarray]] = {}
    time_points = None

    for idx, diameter_um in enumerate(DIAMETER_GRID_UM):
        diameter_m = diameter_um * 1e-6
        times, vel_hist = simulate_velocity_history(
            diameter_m,
            phys_model,
            DEVICE,
            SIMULATION_STEPS,
            RECORD_INTERVAL,
            SAMPLES_PER_DIAMETER,
        )
        velocity_history[idx] = vel_hist
        if time_points is None:
            time_points = times

    # Compute kernel matrices
    kernels, theory = compute_kernel_matrices(DIAMETER_GRID_UM, time_points, velocity_history)

    # Persist data
    np.savez_compressed(
        OUTPUT_DIR / f"kernel_data_freq_{freq_k}k.npz",
        diameters_um=DIAMETER_GRID_UM,
        time_points=time_points,
        kernels=kernels,
        reference_kernel=theory,
    )

    # Plot individual heatmap (still useful for reference)
    plot_kernel_heatmap(DIAMETER_GRID_UM, kernels[-1], theory, frequency)
    
    # Calculate period and filter to one period
    period = 1.0 / frequency
    mask = time_points <= period
    time_filtered = time_points[mask]
    kernels_filtered = kernels[mask]
    avg_rate = kernels_filtered.reshape(len(time_filtered), -1).mean(axis=1)
    
    return time_filtered, avg_rate


def plot_all_frequencies_combined(results: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> None:
    """Plot collision rates for all frequencies in a single figure with smooth curves."""
    
    # Line styles for different frequencies
    line_styles = {
        8000: '-',
        12000: '--',
        16000: '-.',
    }
    
    # Colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(FREQUENCIES)))
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for idx, (freq, color) in enumerate(zip(FREQUENCIES, colors)):
        if freq not in results or results[freq][0] is None:
            continue
            
        time_filtered, avg_rate = results[freq]
        freq_k = freq // 1000
        
        # Convert time to microseconds
        time_us = time_filtered * 1e6
        rate_scaled = avg_rate * 1e12
        
        # Apply smoothing using Gaussian filter
        if len(rate_scaled) > 10:
            rate_smoothed = gaussian_filter1d(rate_scaled, sigma=2.0)
        else:
            rate_smoothed = rate_scaled
        
        # Plot smooth curve
        ax.plot(time_us, rate_smoothed,
                linestyle=line_styles[freq],
                linewidth=2.5,
                color=color,
                label=f'{freq_k} kHz',
                alpha=0.85)
    
    ax.set_xlabel('Time (μs) - One Period', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'Collision Rate [$10^{-12}$ m$^3$/s]', fontsize=14, fontweight='bold')
    ax.set_title('Collision Rate vs Time for Different Frequencies', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(loc='best', fontsize=12, framealpha=0.95, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "collision_rate_all_frequencies.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved combined plot: {out_path}")


def main():
    print("=" * 80)
    print("Collision Rate Analysis - PINN Model")
    print("=" * 80)
    print(f"Frequencies: {[f'{f//1000}k' for f in FREQUENCIES]} Hz")
    print(f"Simulation steps: {SIMULATION_STEPS}, Sampling interval: {RECORD_INTERVAL} μs")
    print("=" * 80)
   
    proceed = "y"
    if os.environ.get("KERNEL_EXTRACTOR_PROMPT", "").strip() == "1":
        proceed = input("\nStart analysis? (y/n): ").strip().lower()
    if proceed != "y":
        print("Aborted.")
        return

    backups = backup_frequency_files()
    results = {}
    
    try:
        print("\nProcessing frequencies...")
        for idx, freq in enumerate(FREQUENCIES):
            time_filtered, avg_rate = process_frequency(freq, idx, len(FREQUENCIES))
            results[freq] = (time_filtered, avg_rate)
        
        print("\n" + "=" * 80)
        print("✓ All frequencies processed!")
        print("=" * 80)
        
        # Generate combined plot
        print("\nGenerating combined frequency plot...")
        plot_all_frequencies_combined(results)
        
        print("\n✓ Analysis complete!")
        print(f"Results saved to: {OUTPUT_DIR}")
        
    finally:
        print("\nRestoring configuration files...")
        restore_frequency_files(backups)
        print("Done.")


if __name__ == "__main__":
    main()

