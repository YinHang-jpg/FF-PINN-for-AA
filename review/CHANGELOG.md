# CHANGELOG

Notable changes to the FF-PINN code accompanying the manuscript.

## [v1.0.0] — manuscript-submission release

### Included

- Five-branch FF-PINN (SPGF x/t + Stokes x/t/v) with Fourier features (`M = 32`).
- Unified runtime wrapper `mechanisms/UNIFIED_PINN.py`.
- Drivers: `other_files/PINN_main.py`, `PINN_main_integrated.py`, `DEM_main.py`.
- Standing-wave initialization and FEM / DEM validation utilities.
- Figure reproduction scripts under `scripts/` and `results/` (outputs in `figures/`).
- Reviewer materials under `review/`.

### Scope

Primary-force surrogates only (SPGF + Stokes). Acoustic wake and related
side studies are excluded from this release.
