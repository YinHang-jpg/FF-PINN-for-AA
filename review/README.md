# `review/` — Supplementary material for paper review

This folder bundles the documents requested by the journal under
*"software implementation and performance details"*, plus a minimal
runnable example. It is intended to make the paper-accompanying
codebase **independently reproducible**.

## Contents

| File | What it covers |
|------|----------------|
| [`SOFTWARE.md`](SOFTWARE.md)             | Implementation details: language, framework, PINN + Fourier-feature architecture, per-module description, input / output specification. |
| [`PERFORMANCE.md`](PERFORMANCE.md)       | Training and inference wall-clock costs, scaling with particle count, comparison with DEM and COMSOL FEM baselines, accuracy / cost trade-off. |
| [`ENVIRONMENT.md`](ENVIRONMENT.md)       | Python version, dependency list, install commands (pip and conda), platform notes. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Step-by-step reproduction of every key manuscript figure, expected wall times, troubleshooting. |
| [`CHANGELOG.md`](CHANGELOG.md)           | Version history; the `v1.0.0` tag corresponds exactly to the manuscript-submission revision. |
| [`examples/`](examples/)                 | Minimal runnable example (`run_example.py`) with a small input file and an expected-output transcript. |

## Recommended reading order

1. [`ENVIRONMENT.md`](ENVIRONMENT.md) → install dependencies.
2. [`examples/run_example.py`](examples/run_example.py) → smoke test.
3. [`SOFTWARE.md`](SOFTWARE.md) → understand the architecture.
4. [`PERFORMANCE.md`](PERFORMANCE.md) → consult the timing tables.
5. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) → re-create the
   manuscript figures.
6. [`CHANGELOG.md`](CHANGELOG.md) → see what changes between
   revisions.

## Quick start

```bash
# from the repository root
pip install -r other_files/requirements.txt
pip install torch scipy tqdm
python review/examples/run_example.py
```

The example wraps `results/PD_vs_time/PD_time.py` to integrate the
PINN model for **0.01 s of physical time** (10 000 steps with
`dt = 1e-6 s`) and writes a single-time-point PINN / DEM / FEM
density-comparison figure to
`review/examples/density_comparison_t0.01s.png` — the same layout
as `results/PD_vs_time/comparison/density_comparison_group1.png`
but showing only the `t = 0.01 s` curves. If the printed lines
match [`examples/expected_output.txt`](examples/expected_output.txt)
within floating-point tolerance and the saved figure shows two
density peaks (`x ≈ 8 mm` and `x ≈ 25 mm`) with the three
pipelines tracking each other, the install is healthy.

Expected wall time: **~ 20 s on GPU** or **~ 3–5 min on CPU**.
