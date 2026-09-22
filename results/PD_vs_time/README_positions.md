# Particle positions for DEM / FEM / FF-PINN comparison (Fig. 8)

`PD_time.py` integrates the unified FF-PINN model and can export particle
positions for comparison against DEM and COMSOL FEM.

## Typical usage

```bash
python results/PD_vs_time/PD_time.py
python results/PD_vs_time/PD_vs_time_DEM.py
python results/PD_vs_time/comparison/PINN_COMSOL_comparison.py
```

Useful outputs:

- `pinn_particle_positions.csv`
- `dem_particle_distribution_data.csv`
- comparison figures under `results/PD_vs_time/comparison/`

FEM reference trajectories are provided at `validation/comsol_positions.csv`
(from `FEM_validation.mph`).
