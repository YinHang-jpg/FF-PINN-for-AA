"""Reproduce manuscript computation-time figures from archived CSV tables.

Reads:
  other_files/computation_time_comparison.csv
  other_files/computation_time_comparision_particle.csv

Writes:
  figures/fig12_computation_time_comparison.png
  figures/fig12_computation_time_vs_particle_count.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results"))
from _figures import FIGURES  # noqa: E402

SIMTIME_CSV = Path(__file__).resolve().parent / "computation_time_comparison.csv"
PARTICLE_CSV = Path(__file__).resolve().parent / "computation_time_comparision_particle.csv"
OUT_SIMTIME = FIGURES / "fig12_computation_time_comparison.png"
OUT_PARTICLE = FIGURES / "fig12_computation_time_vs_particle_count.png"


def read_simtime_csv(csv_path: Path):
    sim_time, pinn, dem, comsol = [], [], [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_time.append(float(row["simulation_time"]))
            pinn.append(float(row["PINN"]))
            dem.append(float(row["DEM"]))
            comsol.append(float(row["COMSOL"]))
    return (
        np.asarray(sim_time, float),
        np.asarray(pinn, float),
        np.asarray(dem, float),
        np.asarray(comsol, float),
    )


def read_particle_csv(csv_path: Path):
    dem_x, dem_y = [], []
    pinn_x, pinn_y = [], []
    comsol_x, comsol_y = [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = (row.get("model") or "").strip().upper()
            pc = int(row["particle_count"].strip())
            ct = float(row["computation_time"].strip())
            if model == "DEM":
                dem_x.append(pc)
                dem_y.append(ct)
            elif model == "PINN":
                pinn_x.append(pc)
                pinn_y.append(ct)
            elif model == "COMSOL":
                comsol_x.append(pc)
                comsol_y.append(ct)
    return (
        np.asarray(dem_x, float),
        np.asarray(dem_y, float),
        np.asarray(pinn_x, float),
        np.asarray(pinn_y, float),
        np.asarray(comsol_x, float),
        np.asarray(comsol_y, float),
    )


def linear_fit(x: np.ndarray, y: np.ndarray):
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    return coeffs, x_line, y_line


def confidence_band(x, y, coeffs, x_line, confidence=0.95):
    n = len(x)
    y_hat = np.polyval(coeffs, x)
    residuals = y - y_hat
    s_e = np.sqrt(np.sum(residuals**2) / (n - 2))
    x_mean = np.mean(x)
    ss_xx = np.sum((x - x_mean) ** 2)
    t_val = t_dist.ppf((1 + confidence) / 2, df=n - 2)
    margin = t_val * s_e * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / ss_xx)
    y_fit = np.polyval(coeffs, x_line)
    return y_fit - margin, y_fit + margin


def plot_series_linear(ax, x, y, coeffs, x_line, y_line, color, marker,
                       marker_size, linestyle, label):
    lo, hi = confidence_band(x, y, coeffs, x_line)
    ax.fill_between(x_line, lo, hi, color=color, alpha=0.15)
    ax.scatter(x, y, color=color, marker=marker, s=marker_size, zorder=5)
    ax.plot(x_line, y_line, color=color, linewidth=2, linestyle=linestyle, label=label)


def _style_axes(ax):
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12)


def plot_simtime(sim_time, pinn, dem, comsol, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    pinn_c, pinn_xl, pinn_yl = linear_fit(sim_time, pinn)
    dem_c, dem_xl, dem_yl = linear_fit(sim_time, dem)
    comsol_c, comsol_xl, comsol_yl = linear_fit(sim_time, comsol)

    plot_series_linear(ax, sim_time, pinn, pinn_c, pinn_xl, pinn_yl,
                       "red", "o", 40, "-", "FF-PINN")
    plot_series_linear(ax, sim_time, dem, dem_c, dem_xl, dem_yl,
                       "green", "^", 50, "-", "DEM")
    plot_series_linear(ax, sim_time, comsol, comsol_c, comsol_xl, comsol_yl,
                       "blue", "s", 40, "--", "FEM")

    ax.set_title("(a)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Simulation Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Computation Time (s)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    _style_axes(ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out_path}")


def plot_particle(dem_x, dem_y, pinn_x, pinn_y, comsol_x, comsol_y, out_path: Path) -> None:
    """Fit and confidence bands on the same (display) x scale: N / 1e6."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scale = 1e6
    # Fit on scaled particle counts so poly coeffs match the plotted x-axis.
    # (Previously coeffs were fit on raw N while bands used N/1e6 → inverted U-bands.)
    pinn_xs, dem_xs = pinn_x / scale, dem_x / scale
    comsol_xs = comsol_x / scale

    pinn_c, pinn_xl, pinn_yl = linear_fit(pinn_xs, pinn_y)
    dem_c, dem_xl, dem_yl = linear_fit(dem_xs, dem_y)

    plot_series_linear(ax, pinn_xs, pinn_y, pinn_c, pinn_xl, pinn_yl,
                       "red", "o", 40, "-", "FF-PINN")
    plot_series_linear(ax, dem_xs, dem_y, dem_c, dem_xl, dem_yl,
                       "green", "^", 50, "-", "DEM")
    if len(comsol_x) >= 3:
        comsol_c, comsol_xl, comsol_yl = linear_fit(comsol_xs, comsol_y)
        plot_series_linear(ax, comsol_xs, comsol_y, comsol_c,
                           comsol_xl, comsol_yl,
                           "blue", "s", 40, "--", "FEM")
    elif len(comsol_x) > 0:
        ax.scatter(comsol_xs, comsol_y, color="blue", marker="s",
                   s=60, zorder=5, label="FEM")

    ax.set_title("(b)", fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Particle Count ($\times 10^{6}$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Computation Time (s)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    _style_axes(ax)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"WROTE {out_path}")


def main() -> None:
    if not SIMTIME_CSV.is_file():
        raise FileNotFoundError(f"Missing timing table: {SIMTIME_CSV}")
    if not PARTICLE_CSV.is_file():
        raise FileNotFoundError(f"Missing timing table: {PARTICLE_CSV}")

    st, pinn, dem, comsol = read_simtime_csv(SIMTIME_CSV)
    plot_simtime(st, pinn, dem, comsol, OUT_SIMTIME)

    dx, dy, px, py, cx, cy = read_particle_csv(PARTICLE_CSV)
    plot_particle(dx, dy, px, py, cx, cy, OUT_PARTICLE)


if __name__ == "__main__":
    main()
