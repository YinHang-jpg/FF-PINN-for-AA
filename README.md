# FF-PINN — Physics-informed neural networks for acoustic particle dynamics

This repository contains PINN models (SPGF / Stokes branches), particle time-stepping drivers, DEM baselines, COMSOL validation utilities, and post-processing for orthokinetic collision kernels referenced in the accompanying manuscript.

## Paper overview (for reviewers)

A **standalone HTML page** summarizes the manuscript’s main points, key results, and how to navigate this repository. Open it in any modern web browser (no web server needed for local viewing).

**[Paper overview (HTML)](front_page/front_page.html)**

When you view this README on a Git hosting site (e.g. GitHub or GitLab), the link above resolves to the normal project URL for that file; use your browser’s **Open / Download / Raw** control if the site does not render HTML inline. This is **not** a `file://` path and does not assume a particular folder on your computer.

## Python environment

- Recommended: **Python 3.10+** (3.9 may work if dependencies install cleanly).
- Install core scientific stack, then **PyTorch** for your platform from [pytorch.org](https://pytorch.org).

```text
pip install -r other_files/requirements.txt
pip install torch scipy tqdm
```

Add the **repository root** to `PYTHONPATH` (or run scripts with the project root as the current working directory) so imports such as `mechanisms.*` and `initialization.*` resolve.

## Layout and main entry points

| Role | Location |
|------|----------|
| Unified decomposition inference (five sub-models) | `other_files/PINN_main.py`, `results/animation/particle_shift.py` |
| Unified **physical** model wrapper | `other_files/clustering.py`, `other_files/PINN_main_integrated.py` |
| DEM baseline | `other_files/DEM_main.py` |
| SPGF / Stokes training scripts | `mechanisms/ARF_PINN_*.py`, `mechanisms/STOKES_PINN_*.py` |
| Runtime unified normalizer + loader | `mechanisms/UNIFIED_PINN.py` |
| COMSOL / density validation | `validation/comsol_visualizer.py`, `validation/comsol_dualplot.py` |
| Parametric SPL / frequency wide model | `PINN_wide/f_SPL_dens_var.py` |
| Orthokinetic kernel post-processing (Fig. 12–style) | `results/kernel/kernel_extractor.py` |
| Frequency / density parameter sweeps | `results/parameter_sweep/freq_sweep.py`, `results/density_sweep/density_distribution_plot.py` |

## Reviewer documentation

- **`docs/REVIEWER_CODE_MAP.md`** — Narrative order: manuscript sections → code paths → data/weights.
- **`docs/PAPER_CODE_CONSISTENCY.md`** — Known alignments and residual naming notes (e.g. kernel script vs. text).


## License / attribution

Use and citation should follow the manuscript and institutional policies. No license file is implied unless one is added separately.
