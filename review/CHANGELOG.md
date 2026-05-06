# CHANGELOG

All notable changes to the **FF-PINN** code accompanying the
manuscript are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/).

---

## [v1.0.0] — 2026-05-05 (manuscript-submission release)

**Status:** This is the version of the code that corresponds
**exactly** to the figures and numerical results reported in the
submitted manuscript (revision 6).

### Added

- Five-branch factorised PINN with random Fourier features:
  `mechanisms/ARF_PINN_x.py`, `ARF_PINN_t.py`,
  `STOKES_PINN_x.py`, `STOKES_PINN_t.py`, `STOKES_PINN_v.py`.
- Unified runtime wrapper `mechanisms/UNIFIED_PINN.py`
  (`PhysicalUnifiedModel`) that loads all five sub-networks and
  combines them into a single `compute_force(x, v_x, t)` call.
- Main entry drivers under `other_files/`:
  - `PINN_main.py` — direct five-network integration.
  - `PINN_main_integrated.py` — integration via the unified
    wrapper.
  - `clustering.py` / `clustering_auto.py` — PINN force stack with
    1-D x-merge collision/coalescence.
  - `DEM_main.py` — analytical Stokes / ARF baseline used in
    Section "Validation".
  - `PINN_preview.py` — light force preview at `t = 0`.
- Standing-wave acoustic source and SPL conversion in
  `initialization/sound_source_standing.py`.
- Frequency-pack trained weights `PINN/freq_*k/` for the sweep
  reported in the paper.
- Post-processing: kernel extraction
  (`results/kernel/kernel_extractor.py`), parameter sweeps
  (`results/parameter_sweep/freq_sweep.py`,
  `results/density_sweep/density_distribution_plot.py`),
  `dt`-convergence study, animation, KDE density tools.
- Validation against COMSOL FEM
  (`validation/comsol_visualizer.py`,
  `validation/comsol_dualplot.py`).
- Timing comparison data and plotting
  (`other_files/computation_time_*.csv`,
  `other_files/plot_time_comparison.py`).
- **Supplementary review material under `review/`:**
  - `SOFTWARE.md` — implementation overview.
  - `PERFORMANCE.md` — runtime tables and FEM comparison.
  - `ENVIRONMENT.md` — install and dependency spec.
  - `REPRODUCIBILITY.md` — step-by-step reproduction guide.
  - `CHANGELOG.md` — this file.
  - `examples/run_example.py`, `examples/demo_input.json`,
    `examples/expected_output.txt` — minimal runnable example.

### Changed

- Integration step `dt` standardised to `1e-6 s` for all paper
  figures (previous experimental drafts used `5e-7 s`).
- `UnifiedNormalizer` now wraps both `x` and `t` periodically inside
  the training window, eliminating Fourier-feature aliasing for
  long integrations.

### Fixed

- Phase convention between Stokes branches (`cos × cos`) made
  consistent across `Stokes_drag.py`, `STOKES_PINN_*.py`, and the
  drivers (matches manuscript Eq. for `u(x, t)`).
- Driver imports normalised so that all entry scripts run from the
  repository root without `sys.path` hacks.

### Removed

- `PINN_wide/` work-in-progress weights (kept locally during
  exploration; not part of the manuscript pipeline).
- `wake_effect/` is **excluded** from the reviewer-facing
  manuscript pipeline (kept in the repository as a side study).

---

## [v0.9.x] — pre-release (internal)

Iterative training of the five sub-networks, normalisation JSON
schema migration (legacy `x_mean / x_std` → compact
`force_mu / force_sigma`), and DEM / COMSOL baselines.
These versions are **not** referenced by the manuscript.

---

## Versioning policy for future revisions

- **Patch (`vMAJOR.MINOR.PATCH`):** documentation, CSV regeneration,
  bug fixes that do not change reported numbers within the printed
  precision.
- **Minor:** new sub-network branch, new sweep, additional
  validation target. Numerical figures may shift within the stated
  error bars.
- **Major:** change of the network factorisation, of the integrator,
  or of the acoustic-source model. The manuscript reference would
  need to be re-pinned to a specific tag.
