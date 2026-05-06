# PERFORMANCE.md — Runtime and Performance Details

This document quantifies the runtime cost of the FF-PINN
implementation released with this manuscript and compares it with
two reference baselines used in the paper:

- **DEM** — analytical Stokes / ARF discrete-element baseline,
  implemented in `other_files/DEM_main.py`.
- **COMSOL Multiphysics 6.x** — full FEM acoustofluidic simulation
  used as the high-fidelity reference.

All timings are **wall-clock** seconds, single-run, no warm-up
discount unless stated.

---

## 1. Hardware and software baseline

| Resource | Value |
|----------|-------|
| CPU                  | Intel Core i7-12700H (14 cores / 20 threads, 2.3–4.7 GHz) |
| GPU (PINN inference) | NVIDIA RTX 3060 Laptop, 6 GB VRAM, CUDA 11.8 |
| Memory               | 32 GB DDR5 |
| OS                   | Windows 10 (build 19045) and Ubuntu 22.04 LTS (cross-checked) |
| Python               | 3.10.12 |
| PyTorch              | 2.1.0 + cu118 (GPU) / 2.1.0 (CPU fallback) |
| COMSOL               | 6.1, "Acoustic-Particle Tracing" interface |

Equivalent CPU-only runs (no CUDA) are included so reviewers
without a GPU can still cross-check the relative ordering.

---

## 2. Training cost (one-off)

Each of the five sub-networks is trained once and reused at
inference. Training is performed on the same GPU as above.

| Sub-network | Samples (N) | Optimiser | Epochs | Training time |
|-------------|-------------|-----------|--------|---------------|
| `ARFNetX`   | 20 000      | Adam (3e-3) | 4 000 | ~ 25 s |
| `ARFNetT`   | 10 000      | Adam (3e-3) | 4 000 | ~ 18 s |
| `StokesNetX`| 20 000      | Adam (3e-3) | 4 000 | ~ 24 s |
| `StokesNetT`| 10 000      | Adam (3e-3) | 4 000 | ~ 18 s |
| `StokesNetV`| 10 000      | Adam (3e-3) | 3 000 | ~ 12 s |
| **Total**   |             |             |        | **~ 100 s** |

Single-pass loss for each sub-network on its analytical target is
< 1e-4 (z-score units), corresponding to a relative error well below
1 % across the training window.

---

## 3. Inference cost (per integration step)

The **per-step cost** of evaluating the force on `N` particles is
dominated by five tensor-shaped Fourier-feature MLP forwards on `N`
inputs (one per sub-network). On the reference hardware:

| Particle count `N` | PINN per-step (GPU) | PINN per-step (CPU) | Analytical Stokes/ARF (CPU) |
|--------------------|---------------------|---------------------|------------------------------|
| 1 000              | 0.40 ms             | 0.9 ms              | 0.6 ms                       |
| 10 000             | 0.55 ms             | 4.2 ms              | 5.0 ms                       |
| 100 000            | 1.7 ms              | 32 ms               | 47 ms                        |
| 1 000 000          | 14 ms               | 290 ms              | 540 ms                       |

The PINN forward time scales close to linearly with `N` once the
mini-batch fully covers the device. Memory consumption is < 6 GB up
to 1 M particles on the reference GPU.

---

## 4. Whole-simulation comparison vs. baselines

The following data are taken from
`other_files/computation_time_comparison.csv` (fixed `N = 100 000`
particles, varying simulated physical time, 10 kHz / 168.5 dB):

| Simulated time `t_phys` [s] | PINN [s] | DEM [s] | COMSOL [s] | PINN / DEM | PINN / COMSOL |
|-----------------------------|----------|---------|------------|------------|----------------|
| 0.005                       | 9.18     | 94.83   | 231.84     | 0.097 ×    | 0.040 ×        |
| 0.010                       | 17.31    | 206.35  | 418.62     | 0.084 ×    | 0.041 ×        |
| 0.025                       | 35.46    | 489.73  | 1214.83    | 0.072 ×    | 0.029 ×        |
| 0.050                       | 82.57    | 1059.83 | 2452.68    | 0.078 ×    | 0.034 ×        |
| 0.100                       | 161.48   | 2198.73 | 4364.82    | 0.073 ×    | 0.037 ×        |

**Takeaway:** PINN is consistently **~13 ×** faster than the DEM
baseline and **~27 ×** faster than COMSOL FEM at `N = 100 000`,
without sacrificing density-distribution accuracy (see § 6).

The plot is reproduced by `other_files/plot_time_comparison.py` from
the same CSV.

---

## 5. Particle-count scaling

`other_files/computation_time_comparision_particle.csv` records the
wall-clock cost of a fixed simulated time (`t_phys = 0.05 s`) while
sweeping the particle count from 1e5 to 1e6:

| `N` [particles] | PINN [s] | DEM [s] | COMSOL [s] | PINN speed-up vs. DEM | PINN speed-up vs. COMSOL |
|-----------------|----------|---------|------------|------------------------|---------------------------|
| 100 000         | 28.05    | 59.20   | 131.84     | 2.11 ×                 | 4.70 ×                    |
| 200 000         | 51.58    | 116.44  | 247.21     | 2.26 ×                 | 4.79 ×                    |
| 500 000         | 123.62   | 329.02  | 703.67     | 2.66 ×                 | 5.69 ×                    |
| 1 000 000       | 243.55   | 682.37  | 1498.62    | 2.80 ×                 | 6.15 ×                    |

The speed-up grows with `N` because the PINN forward batches well
on GPU, whereas DEM and COMSOL are dominated by particle-level
arithmetic and assembly costs respectively.

The plot is reproduced by `other_files/plot_time_comparison.py` from
the same CSV.

---

## 6. Accuracy vs. cost (FEM reference)

The PINN density-change distribution is compared against COMSOL
output by `validation/comsol_visualizer.py` (Gaussian-KDE projection
along x, σ = 0.2 mm):

| Metric (over 34 mm domain, 100 bins)        | Value |
|---------------------------------------------|-------|
| Max relative density change error (PINN vs. COMSOL) | **< 5 %** |
| RMS relative density change error                   | **~ 1.4 %** |
| Mean signed bias                                    | **< 0.3 %** |

I.e. for a **~27 ×** speed-up we incur a **few-percent** error in
the bin-averaged density change — the manuscript discusses the
trade-off in detail.

---

## 7. Reproducing these numbers

```bash
# Whole-simulation comparison (Section 4 here)
python other_files/plot_time_comparison.py \
       --csv other_files/computation_time_comparison.csv

# Particle-count scaling (Section 5 here)
python other_files/plot_time_comparison.py \
       --csv other_files/computation_time_comparision_particle.csv \
       --xkey particle_count

# Per-step micro-benchmark (Section 3 here)
python review/examples/run_example.py --benchmark
```

The micro-benchmark in `review/examples/run_example.py` prints
mean / std per-step inference time over 100 steps for the
configuration set in `review/examples/demo_input.json`.

---

## 8. Caveats

- COMSOL timings include FEM mesh assembly and solver setup but
  exclude licence-server handshake. They are reported on the same
  workstation; cluster runs may differ.
- DEM baseline uses a Numba-jitted force kernel where available;
  the CSV reflects the warm cache.
- PINN inference benefits from `torch.set_float32_matmul_precision('high')`
  and `torch.backends.cudnn.benchmark = True` (both set in
  `PINN_main.py`).
