# `other_files/` — primary simulation drivers and utilities

This folder hosts **main-entry** style scripts and small tools. The historical root-level `main.py` described in older notes is **not** the current layout; use the paths below.

## Main entry scripts

| Script | Purpose |
|--------|---------|
| `PINN_main.py` | Time marching with **five trained sub-PINNs** (ARF x/t + Stokes x/t/v); no live animation. |
| `PINN_main_integrated.py` | Uses **`PhysicalUnifiedModel`** (`UNIFIED_PINN`) with weights under `PINN/`. |
| `clustering.py` | Same force stack as integrated path plus **1D x-merge** collision coalescence and diagnostics. |
| `DEM_main.py` | Discrete-element style baseline importing `mechanisms.ARF.compute_pressure_gradient_and_apply_arf`. |
| `PINN_preview.py` | Lightweight preview of forces on a particle layout. |

**Encoding:** several scripts call `sys.stdout.reconfigure(encoding="utf-8")` (or equivalent) on Windows so log text stays readable.

## Dependencies

See `other_files/requirements.txt` for a **minimal** list. Training and inference also need **PyTorch**, **SciPy**, and typically **tqdm** — install separately for your CUDA/CPU stack.

## Project root / `PYTHONPATH`

Run these scripts with the **repository root** as the working directory (parent of `mechanisms/`), or export:

```text
set PYTHONPATH=<path-to-repo-root>
```

so `import mechanisms` and `import initialization` succeed.

## Models

Trained weights are expected under `PINN/` at the repo root, often in subfolders such as `PINN/freq_8k/`. Some drivers honor `PINN_FREQ_FOLDER` to pick a frequency-specific set.

## Full reviewer map

See `docs/REVIEWER_CODE_MAP.md` at the repository root.
