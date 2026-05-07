# REPRODUCIBILITY.md — Reproducibility Guide

This guide enables an independent reviewer to reproduce, **from a
clean clone**, every key numerical and figure-level result reported
in the manuscript. Commands assume a POSIX shell; the equivalent
Windows commands are noted where relevant.

---

## 1. From zero to first plot (≈ 5 minutes)

```bash
# 1. Clone and enter the repository
git clone https://github.com/YinHang-jpg/FF-PINN
cd FF-PINN

# 2. Set up the Python environment (see review/ENVIRONMENT.md)
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
pip install -r other_files/requirements.txt
pip install torch scipy tqdm

# 3. Make sub-packages importable from the repo root
export PYTHONPATH=$(pwd)                     # Windows (PS): $env:PYTHONPATH = (Get-Location).Path

# 4. Smoke-test the install with the minimal demo
python -m review.examples.run_example
```

The demo wraps `results/PD_vs_time/PD_time.py`'s
`run_single_simulation` for **0.01 s of physical time** (10 000
timesteps with `dt = 1e-6 s`), then loads the bundled DEM and FEM
(COMSOL) reference trajectories and writes a single-time-point
PINN / DEM / FEM density-comparison figure
`review/examples/density_comparison_t0.01s.png` (same style as the
manuscript figure
`results/PD_vs_time/comparison/density_comparison_group1.png` but
showing only the `t = 0.01 s` curves). If the printed lines match
[`examples/expected_output.txt`](examples/expected_output.txt) and
the figure shows two density peaks at `x ≈ 8 mm` and `x ≈ 25 mm`
with PINN / DEM / FEM in agreement, the install is healthy.

---

## 2. Reproduce the manuscript figures step-by-step

### 2.1 Figure: PINN vs. analytical force fields (Section "SPGF")

```bash
# Spatial branch (cos(kx)) — figure (a) class
python scripts/plot_Fxt_SPGF.py

# Stokes branches — figure (b) class
python scripts/plot_Fxt_Stokes.py

# Marginals along x and t (paper inset)
python scripts/plot_x.py
python scripts/plot_t.py
```

These reuse the trained weights under `PINN/` (root folder, default
10 kHz pack). Expected wall time: **< 30 s** on a laptop CPU.

### 2.2 Figure: Particle density vs. time (Section "Validation")

```bash
python other_files/PINN_main.py
```

By default this advances `N = 10 000` particles for `1000 × dt`
(`dt = 1e-6 s`) on the standing-wave field at 10 kHz / 168.5 dB
and pops up the dual-panel "scatter + density" matplotlib figure.
Expected wall time: **~ 30 s** (CPU) or **~ 5 s** (GPU).

To reproduce the **paper-resolution** version with `1e5`
particles and 5 ms physical time:

```bash
# Edit `steps` and `N` at the top of PINN_main.py, then:
python other_files/PINN_main.py
```

Expected wall time on the reference workstation:
**~ 90 s** (GPU) / **~ 20 min** (CPU).

### 2.3 Figure: Density distribution vs. COMSOL

```bash
python validation/comsol_visualizer.py    \
       --pinn-positions <PINN run output> \
       --comsol-csv validation/comsol_positions.csv
```

(Default arguments inside the script point to the bundled
`comsol_positions.csv` and a precomputed PINN positions file. Run
without flags for the bundled comparison.)

### 2.4 Figure: Computation time comparisons

```bash
# Wall time vs. simulated time
python other_files/plot_time_comparison.py \
       --csv other_files/computation_time_comparison.csv

# Wall time vs. particle count
python other_files/plot_time_comparison.py \
       --csv other_files/computation_time_comparision_particle.csv \
       --xkey particle_count
```

Outputs the figures
`other_files/computation_time_comparison.png` and
`other_files/computation_time_vs_particle_count.png` (already
included in the repository as a reference).

### 2.5 Figure: Orthokinetic collision kernel (Fig. 12-class)

```bash
python results/kernel/kernel_extractor.py
```

Writes `.npz` archives and PNG heat maps under
`results/kernel/`. Expected wall time: **~ 5 min** on CPU.

### 2.6 Figure: dt convergence

```bash
python results/convergence_check/convergence_check.py
```

Sweeps several `dt` values through `PhysicalUnifiedModel` and
overlays the resulting density-change curves. Expected wall time:
**~ 3 min** on GPU.

### 2.7 Frequency / density sweeps

```bash
# Frequency sweep — paper Fig. class
python results/parameter_sweep/freq_sweep.py

# Density sweep — paper Fig. class
python results/density_sweep/density_distribution_plot.py
```

Both expect frequency-specific weight packs under `PINN/freq_*k/`.
Use only the frequencies whose pack is present in your checkout.

---

## 3. Recommended one-line driver

For reviewers who only want to confirm the headline numbers
(PINN integration over 0.01 s and a single-time-point comparison
with DEM and FEM):

```bash
python review/examples/run_example.py --plot
```

This prints:

- confirmation that weights load from the repository `PINN/` folder
  (default; optional `pinn_model_base_path` in `demo_input.json` for
  advanced overrides),
- snapshot times retained from the integrator (`[0.0, 0.01]`),
- relative-concentration-change ranges at `t = 0.01 s` for the
  PINN, DEM and FEM pipelines, and
- pointwise RMSE / MAE between the PINN and FEM profiles over the
  `[2 mm, 32 mm]` analysis window.

It also writes
`review/examples/density_comparison_t0.01s.png` (a single-time
version of `density_comparison_group1.png`).

---

## 4. Expected wall times

| Stage                                 | Hardware        | Expected time |
|---------------------------------------|-----------------|---------------|
| `review/examples/run_example.py`      | Laptop GPU      | ~ 20 s        |
| `review/examples/run_example.py`      | Laptop CPU      | ~ 3–5 min     |
| `other_files/PINN_main.py` (defaults) | Laptop CPU      | ~ 30 s        |
| Re-train one sub-network              | Laptop GPU      | ~ 25 s        |
| Re-train all five sub-networks        | Laptop GPU      | ~ 100 s       |
| §2.2 paper-resolution density figure  | Laptop GPU      | ~ 90 s        |
| §2.4 timing-comparison plots          | Any             | < 5 s         |
| §2.5 kernel extractor                 | Laptop CPU      | ~ 5 min       |
| §2.6 dt convergence                   | Laptop GPU      | ~ 3 min       |

---

## 5. Determinism

- All NumPy randomness is seeded with `np.random.default_rng(seed)`
  inside `initialization/particle_initialization.py`. Default
  seed: `0`.
- All PyTorch randomness used at training time is seeded by
  `torch.manual_seed(0)` (see the `main()` function of each
  `*_PINN_*.py` file).
- Inference is deterministic up to floating-point reduction order
  on a fixed device. Cross-device reproduction (CPU ↔ GPU) is
  expected to agree to ~ 5 significant digits.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ModuleNotFoundError: mechanisms` | `PYTHONPATH` not set, or running script from a sub-folder | run from the repo root or `export PYTHONPATH=$(pwd)` |
| `FileNotFoundError: PINN/...pth` | Trained weights missing in the checkout | run `review/examples/run_example.py` (uses analytical fallback), or train via `mechanisms/*_PINN_*.py main()` |
| CUDA out-of-memory at `N = 1e6` | Small VRAM | drop to `N = 5e5`, or use CPU (`CUDA_VISIBLE_DEVICES=""`) |
| Plots do not show on a headless server | No display | replace `plt.show()` with `plt.savefig("out.png")` (one-line patch in each driver) |
| Inconsistent SPL or frequency | Wrong global config | edit `initialization/sound_source_standing.py` and rerun training where applicable |

---

## 7. Where to find authoritative results

- **Bundled reference plots** — pre-rendered PNGs under
  `results/` and `other_files/` (e.g.
  `other_files/computation_time_comparison.png`) act as
  ground-truth comparison images.
- **CSV ground truth** — `other_files/computation_time_*.csv`,
  `validation/comsol_positions.csv`,
  `other_files/concentration_curve.txt`.
- **Numerical baselines** — see [`PERFORMANCE.md`](PERFORMANCE.md)
  for the consolidated tables.
