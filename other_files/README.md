# `other_files/` — simulation drivers

Main entry points for particle time-stepping. Mechanisms live in `mechanisms/`,
shared acoustic parameters in `initialization/`, and trained weights in `PINN/`.

| Script | Purpose |
|--------|---------|
| `PINN_main.py` | Time marching with the five trained sub-networks |
| `PINN_main_integrated.py` | Time marching via `PhysicalUnifiedModel` |
| `DEM_main.py` | Analytical SPGF + Stokes DEM baseline |
| `plot_time_comparison.py` | Manuscript Fig. 12 timing plots (from CSV tables below) |
| `computation_time_comparison.csv` | Wall-clock vs simulated time (`N = 100 000`) |
| `computation_time_comparision_particle.csv` | Wall-clock vs particle count (`t = 0.05 s`) |
| `run_density_sweep.py` | Launcher for `results/density_sweep` |
| `requirements.txt` | Minimal scientific-stack dependencies |

Run scripts from the repository root (or set `PYTHONPATH` to the root).
