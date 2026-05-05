# Reviewer code map (FF-PINN)

## Repository layout (four pillars)

| Folder | Contents |
|--------|----------|
| **`mechanisms/`** | ARF and Stokes **physics + PINN** definitions, `UNIFIED_PINN.py` runtime, training scripts for the five sub-networks. |
| **`initialization/`** | Particle layouts, standing-wave parameters (`sound_source_standing.py`), shared **inputs** to mechanisms and drivers. |
| **`PINN/`** | Trained **`.pth` + normalization JSON**; often organized as `PINN/freq_8k/`, `freq_12k/`, `freq_16k/`, etc. |
| **`results/`** | **Outputs**: kernel extraction, parameter sweeps, animations, convergence studies, efficiency tables. |

Scripts under `other_files/`, `scripts/`, and `validation/` orchestrate these directories.

Paper section numbering should be checked against your final Word/PDF; subsection titles below follow the usual structure for this project.

---

## Title / abstract / conceptual figure (PINN + physics coupling)

**Narrative:** Unified network decomposition (ARF × phase factors + Stokes drag factors) and time integration.

**Code:**

- Core architecture and normalization: `mechanisms/UNIFIED_PINN.py`
- Sub-network definitions and training: `mechanisms/ARF_PINN_x.py`, `ARF_PINN_t.py`, `mechanisms/STOKES_PINN_x.py`, `STOKES_PINN_t.py`, `STOKES_PINN_v.py`
- Acoustic pressure / ARF helpers: `mechanisms/ARF.py`, `initialization/sound_source_standing.py`
- Stokes / Cunningham helpers: `mechanisms/Stokes_drag.py`

**Drivers / demos:**

- Direct batch integration (decomposed models): `other_files/PINN_main.py`
- Clustering + collision merge (example): `other_files/clustering.py`
- Animated triple panel: `results/animation/particle_shift.py`

**Weights:** under `PINN/` (often `PINN/freq_{8,12,16}k/` after frequency sweeps). Exact filenames: `*_model_*.pth` + `*_normalization_params.json` (see `UNIFIED_PINN.load_individual_models`).

---

## SPGF (spatial Fourier features) and Stokes branch plots

**Code:** `scripts/plot_Fxt_SPGF.py`, `scripts/plot_Fxt_Stokes.py`, `scripts/plot_x.py`, `scripts/plot_t.py`

**Depends on:** trained weights + normalization JSON in `PINN/`.

---

## DEM baseline

**Code:** `other_files/DEM_main.py`, related utilities (`other_files/autoclustering_DEM.py`, etc.)

---

## FEM / COMSOL validation and number-density figures

**Code:**

- `validation/comsol_visualizer.py` — KDE-style concentration along x, dual-panel plots
- `validation/comsol_dualplot.py` — multi-time concentration curves from CSV
- `validation/calculation_visualizer.py` — CLI density from `.csv` / `.npy`

**Example data:** `validation/comsol_positions.csv`

---

## Acoustic knobs (SPL, frequency, domain)

**Code / config:** `initialization/sound_source_standing.py` (SPL, frequency, domain).  
Sweep drivers that edit these in place for batch studies live under `results/parameter_sweep/` and `results/density_sweep/`.

---

## Computation time comparisons (e.g. timing figure class)

**Code:** `other_files/plot_time_comparison.py`  
**Data:** `other_files/computation_time_comparison.csv` (and `*_particle` variants if present)

---

## §3.3 Orthokinetic collision kernel / Fig. 12 class

**Code:** `results/kernel/kernel_extractor.py`

**Companion note:** `docs/PAPER_CODE_CONSISTENCY.md` (formulas, frequency list, `DIAMETER_ENH_*` vs. prose “wake” wording).

**Output artifacts:** heat maps, kernel–time curves, `.npz` archives under `results/kernel/`.

---

## Timestep convergence (side study)

**Code:** `results/convergence_check/convergence_check.py` — compares Gaussian-KDE density-change curves for several `dt` choices using `PhysicalUnifiedModel`.

---

## Parameter sweeps (side studies)

- **Frequency:** `results/parameter_sweep/freq_sweep.py`
- **Density:** `results/density_sweep/density_distribution_plot.py`

---

## Excluded from this map

**`wake_effect/`** — optional wake PINN submodule; not part of the reviewer-facing manuscript mapping for this repository revision. See root `README.md`.
