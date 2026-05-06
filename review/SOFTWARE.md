# SOFTWARE.md — Implementation Details

This document describes the software implementation of **FF-PINN**
(Physics-informed Neural Network with Fourier Features) for acoustic
particle dynamics, as accompanying material for the manuscript.

---

## 1. Programming language and frameworks

| Layer | Stack |
|-------|-------|
| Core language        | Python 3.10+ (3.9 may also work) |
| Deep-learning engine | PyTorch (>= 1.13, CPU or CUDA) |
| Numerics             | NumPy, SciPy (`scipy.interpolate.PchipInterpolator`, `scipy.spatial.distance`) |
| Plotting             | Matplotlib |
| Acceleration (optional) | Numba, Joblib (used by some DEM utilities) |
| Misc.                | tqdm, psutil, json |

The PINN training and inference code is **pure PyTorch**. No private
backends, custom C/CUDA kernels, or proprietary solver libraries are
required. CUDA is auto-detected at runtime; if no GPU is found, the
code falls back to CPU without any code change.

---

## 2. Core algorithm: PINN with Fourier features

The full driving force on a micrometer-scale particle in a 1-D
standing acoustic wave is decomposed into two physical contributions
and **factorised** into one-input sub-networks:

```
F_total(x, v_x, t) =  F_ARF(x, t)            + F_Stokes(x, v_x, t)
                   ≈ N_ARFx(x) · N_ARFt(t)   + (-c_d v_x + c_d u(x, t))
                   ≈ N_ARFx(x) · N_ARFt(t)   + (-c_d N_Sv(v_x))
                                              + s · N_Sx(x) · N_St(t)
```

where:

- `c_d = 3 π μ d_p / C_c` — Stokes drag coefficient (Cunningham
  corrected).
- `s = c_d · 2π f · A` — Stokes-driven scaling for the local fluid
  velocity term, with `A` derived from the SPL.
- `u(x,t) = -2π f A · cos(k x) · cos(ω t)` — fluid velocity for a
  standing plane wave.

Each one-input sub-network uses **random Fourier features** (RFF) at
the acoustic wavenumber `k = 2π / λ` (or angular frequency `ω`) to
encode the periodicity of the field, followed by a small MLP:

```
input  →  [sin(W·x), cos(W·x)]  →  Linear(64) → tanh
                                  → Linear(32) → tanh
                                  → Linear(1)
```

with `W ~ N(0, k^2)` for the spatial branches and `W ~ N(0, ω^2)` for
the temporal branches. The MLPs are intentionally small (~3.5k
parameters each) so that the inference cost scales nearly linearly
with the number of particles, while still reproducing the analytical
target in `[-1, 1]` accuracy band on the training window.

Time integration uses an explicit symplectic-Euler scheme with a
fixed small step `dt = 1e-6 s`:

```
F_n         = F(x_n, v_n, t_n)               # PINN forward (one or five sub-models)
v_{n+1}     = v_n + (F_n / m) · dt
x_{n+1}     = x_n + v_{n+1} · dt
```

This is implemented identically in the analytical baseline and in
the PINN driver, so any difference in the trajectories is purely due
to the surrogate force model.

---

## 3. Code modules

### 3.1 Top-level layout (paper-relevant subset)

```
FF-PINN/
├── mechanisms/        # Physics + PINN definitions and training scripts
├── initialization/    # Particle layouts and acoustic source parameters
├── PINN/              # Trained checkpoints (.pth) and normalization JSON
├── other_files/       # Main entry drivers and utilities
├── results/           # Post-processing, figure generation, sweeps
├── validation/        # COMSOL / FEM comparison utilities
├── docs/              # Reviewer documentation (code map, consistency note)
├── review/            # Supplementary material for paper review (this folder)
└── README.md
```

### 3.2 `mechanisms/` — physics and learning models

| File | Role |
|------|------|
| `ARF.py`              | Analytical acoustic radiation force on a sphere from front/back pressure on the standing-wave field. |
| `Stokes_drag.py`      | Stokes drag coefficient (with Cunningham correction) and analytical fluid velocity. |
| `agglomeration.py`    | Pairwise collision / coalescence helper used by the clustering driver. |
| `wake.py`             | Optional wake-effect helper (excluded from the manuscript pipeline). |
| `ARF_PINN_x.py`       | `ARFNet` (Fourier-feature MLP, x-input) + training loop targeting `cos(kx)`. |
| `ARF_PINN_t.py`       | `ARFNetT` (Fourier-feature MLP, t-input) targeting `cos(ωt)`. |
| `STOKES_PINN_x.py`    | `StokesNetX` x-factor sub-network. |
| `STOKES_PINN_t.py`    | `StokesNetT` t-factor sub-network. |
| `STOKES_PINN_v.py`    | `StokesNetV` velocity-factor sub-network. |
| `UNIFIED_PINN.py`     | `PhysicalUnifiedModel` — combines the five sub-networks into one runtime force model with a single `UnifiedNormalizer`. |

### 3.3 `initialization/` — shared physical inputs

| File | Role |
|------|------|
| `particle_initialization.py` | Random non-overlapping particle placement, mass / radius assignment. |
| `sound_source_standing.py`   | Standing-wave parameters (`frequency`, `sound_pressure_level`) and pressure-field grid. |
| `sound_source_traveling.py`  | Traveling-wave variant (sanity checks; not used in the main paper). |
| `test_particle.py`           | Minimal initialization sanity test. |

### 3.4 `PINN/` — trained weights

For each frequency pack (default at the repo root, plus
`PINN/freq_<f>k/` for sweeps):

```
arf_model_x.pth                    arf_model_x_normalization_params.json
arf_model_t.pth                    arf_model_t_normalization_params.json
stokes_model_x.pth                 stokes_model_x_normalization_params.json
stokes_model_t.pth                 stokes_model_t_normalization_params.json
stokes_model_v.pth                 stokes_model_v_normalization_params.json
```

Each `.json` records the input window (`x_min`, `x_max`, `t_min`,
`t_max` or `vx_min`, `vx_max`) and the output `force_mu / force_sigma`
used to undo the z-score during inference. `UNIFIED_PINN.py` reads all
five JSONs with `UnifiedNormalizer.load_from_individual_models`.

### 3.5 `other_files/` — main entry drivers

| File | Role |
|------|------|
| `PINN_main.py`              | Time-marches particles using the **five trained sub-PINNs** directly (no live animation). Reference driver for the manuscript. |
| `PINN_main_integrated.py`   | Same physics, but routes the call through `PhysicalUnifiedModel` (`UNIFIED_PINN`). Useful when a single forward call is desired. |
| `clustering.py`             | PINN force stack + 1-D x-merge collision/coalescence and diagnostics. |
| `clustering_auto.py`        | Driver loop for sweeping parameters of the clustering script. |
| `DEM_main.py`               | Discrete-element-style baseline that imports the analytical ARF / Stokes implementations. |
| `PINN_preview.py`           | Lightweight PINN-vs-analytical force preview at `t = 0`. |
| `plot_time*.py`             | Plotting helpers for timing CSVs. |

### 3.6 `results/` and `validation/` — post-processing

- `results/kernel/kernel_extractor.py` — orthokinetic collision
  kernel extraction (Fig. 12-class outputs).
- `results/parameter_sweep/freq_sweep.py`,
  `results/density_sweep/density_distribution_plot.py` — frequency
  and density sweeps that wrap the main driver.
- `results/convergence_check/convergence_check.py` — `dt`
  convergence study using the unified model.
- `validation/comsol_visualizer.py`,
  `validation/comsol_dualplot.py` — KDE comparison and dual-panel
  plots against COMSOL CSV exports.

### 3.7 `review/examples/` — minimal example (this material)

A self-contained, weights-free demo. See
[`examples/run_example.py`](examples/run_example.py).

---

## 4. Input / output specification

### 4.1 Inputs

**At training time** (`mechanisms/*_PINN_*.py`):

| Quantity | Shape | Description |
|----------|-------|-------------|
| `x` | `(N, 1)` | Particle x-position [m]. Sampled in `[-3λ/2, 3λ/2]` for spatial branches. |
| `t` | `(N, 1)` | Time within one period [s]. |
| `v_x` | `(N, 1)` | Particle x-velocity [m/s]. |
| `F_target` | `(N, 1)` | Analytical target value of the dimensionless factor (e.g. `cos(kx)`). |

Targets are normalised by `(force_mu, force_sigma)` and persisted in
the normalization JSON.

**At inference time** (drivers under `other_files/`):

| Quantity | Shape | Description |
|----------|-------|-------------|
| Particle positions | `(N, 2)` | `(x, y)` in metres, although only the x-channel is used for forces. |
| Particle velocities | `(N, 2)` | `(v_x, v_y)` in m/s. |
| Particle radii | `(N,)` | Particle radius [m]. |
| Particle masses | `(N,)` | Particle mass [kg]. |
| Time scalar `t` | float | Current simulation time [s]. |
| Acoustic config | constants | `frequency` (Hz), `sound_pressure_level` (dB), `domain_size` (m). |

### 4.2 Outputs

| Driver | Output |
|--------|--------|
| `PINN_main.py`             | Final positions, density change vs. uniform baseline (matplotlib figure), wall-clock time per step. |
| `PINN_main_integrated.py`  | Same as above using `PhysicalUnifiedModel`. |
| `clustering.py`            | Particle counts and cluster-merge diagnostics. |
| `kernel_extractor.py`      | `.npz` archive + heat maps and kernel–time curves. |
| `freq_sweep.py` etc.       | Multi-panel figures and CSV summaries under `results/`. |

All numeric outputs are **deterministic** for a fixed seed (set in
`PINN_main.py` / drivers via `numpy.random` and `torch.manual_seed`
where applicable) and a fixed device.

---

## 5. Numerical conventions

- Single-precision (`torch.float32`) is used for both PINN inference
  and integration; this is sufficient given the scale of the
  per-step displacements.
- Periodic input wrapping along x (one wavelength) and along t (one
  period) is handled inside `UnifiedNormalizer.normalize_inputs` so
  that the network never sees out-of-range Fourier-feature inputs.
- All physical quantities are SI. Lengths are reported in mm in
  plots only.
