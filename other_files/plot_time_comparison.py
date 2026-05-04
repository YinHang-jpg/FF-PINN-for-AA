import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# ── Plot 1: fixed particle count, varying simulation time ──────────────

def read_simtime_csv(csv_path):
    sim_time, pinn, dem, comsol = [], [], [], []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim_time.append(float(row["simulation_time"]))
            pinn.append(float(row["PINN"]))
            dem.append(float(row["DEM"]))
            comsol.append(float(row["COMSOL"]))
    return (np.array(sim_time), np.array(pinn),
            np.array(dem), np.array(comsol))


def linear_fit(x, y):
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    return coeffs, x_line, y_line


def confidence_band(x, y, coeffs, x_line, confidence=0.95):
    from scipy.stats import t as t_dist
    n = len(x)
    y_hat = np.polyval(coeffs, x)
    residuals = y - y_hat
    s_e = np.sqrt(np.sum(residuals ** 2) / (n - 2))
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
    ax.plot(x_line, y_line, color=color, linewidth=2,
            linestyle=linestyle, label=label)


def plot_simtime(sim_time, pinn, dem, comsol):
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

    ax.set_title("(a)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Simulation Time (s)", fontsize=12)
    ax.set_ylabel("Computation Time (s)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out = Path(__file__).with_name("computation_time_comparison.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Plot 2: fixed steps, varying particle count ───────────────────────

def read_particle_csv(csv_path):
    dem_x, dem_y = [], []
    pinn_x, pinn_y = [], []
    comsol_x, comsol_y = [], []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
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
    return (np.array(dem_x), np.array(dem_y),
            np.array(pinn_x), np.array(pinn_y),
            np.array(comsol_x), np.array(comsol_y))


def plot_particle(dem_x, dem_y, pinn_x, pinn_y, comsol_x, comsol_y):
    fig, ax = plt.subplots(figsize=(10, 6))

    pinn_c, pinn_xl, pinn_yl = linear_fit(pinn_x, pinn_y)
    dem_c, dem_xl, dem_yl = linear_fit(dem_x, dem_y)

    plot_series_linear(ax, pinn_x, pinn_y, pinn_c, pinn_xl, pinn_yl,
                       "red", "o", 40, "-", "FF-PINN")
    plot_series_linear(ax, dem_x, dem_y, dem_c, dem_xl, dem_yl,
                       "green", "^", 50, "-", "DEM")
    if len(comsol_x) >= 3:
        comsol_c, comsol_xl, comsol_yl = linear_fit(comsol_x, comsol_y)
        plot_series_linear(ax, comsol_x, comsol_y, comsol_c,
                           comsol_xl, comsol_yl,
                           "blue", "s", 40, "--", "FEM")
    elif len(comsol_x) > 0:
        ax.scatter(comsol_x, comsol_y, color="blue", marker="s",
                   s=60, zorder=5, label="FEM")

    ax.set_title("(b)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Particle Count", fontsize=12)
    ax.set_ylabel("Computation Time (s)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    out = Path(__file__).with_name("computation_time_vs_particle_count.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = Path(__file__).parent

    simtime_csv = base / "computation_time_comparison.csv"
    if simtime_csv.exists():
        st, pinn, dem, comsol = read_simtime_csv(simtime_csv)
        plot_simtime(st, pinn, dem, comsol)
    else:
        print(f"Skipped (not found): {simtime_csv}")

    particle_csv = base / "computation_time_comparision_particle.csv"
    if particle_csv.exists():
        dx, dy, px, py, cx, cy = read_particle_csv(particle_csv)
        plot_particle(dx, dy, px, py, cx, cy)
    else:
        print(f"Skipped (not found): {particle_csv}")
