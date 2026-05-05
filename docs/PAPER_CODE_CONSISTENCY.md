# Manuscript vs. code consistency notes

This note helps reviewers align symbols in the paper with this repository **without** opening the optional `wake_effect/` submodule (acoustic wake PINNs used elsewhere in the project).

## §3.3 orthokinetic kernel (`results/kernel/kernel_extractor.py`)

- **Frequencies:** The kernel driver loops over `FREQUENCIES = [8000, 12000, 16000]` Hz. For each value it temporarily copies the matching checkpoints from `PINN/freq_{8,12,16}k/` into the active `PINN/` root, updates `frequency` in `initialization/sound_source_standing.py` and listed mechanism files, then restores backups.

- **Collision adhesion probability:** The code uses  
  \(P_{\mathrm{coll}}(|\Delta v|) = 1 - \exp(-|\Delta v|/v_s)\) with `RELATIVE_SPEED_SCALE = 0.02` m/s (`v_s` in the manuscript).

- **Diameter scaling (not wake physics):** `DIAMETER_ENH_COEFF`, `DIAMETER_ENH_EXP`, and `REF_DIAMETER_M` implement a **heuristic polydispersity factor** applied to the sum of diameters when converting velocity statistics to a kernel. This is **not** the particle wake model trained under `wake_effect/`; the comment in code explicitly distinguishes the two.

- **Outputs:** Heat maps, reference-kernel comparison panels, collision-rate curves, and `.npz` archives are written under `results/kernel/`.

## Four-way layout (reviewer navigation)

| Role | Directory |
|------|-----------|
| Closed-form + PINN **mechanisms** referenced in the text | `mechanisms/` |
| Initial conditions, sound field, global **parameters** | `initialization/` |
| Trained **weights** (`.pth` + normalization JSON) | `PINN/` (often `PINN/freq_*k/`) |
| Figures, sweeps, kernel post-processing **results** | `results/` |

Drivers in `other_files/` and `scripts/` compose these pieces at runtime.
