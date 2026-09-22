# SOFTWARE.md — Implementation Details

Software implementation of **FF-PINN** (Fourier-featured Physics-informed Neural
Network) for acoustic agglomeration under SPGF and Stokes drag.

---

## 1. Language and frameworks

| Layer | Stack |
|-------|-------|
| Language | Python 3.10+ |
| Deep learning | PyTorch (>= 1.13, CPU or CUDA) |
| Numerics | NumPy, SciPy |
| Plotting | Matplotlib |
| Misc. | tqdm, pandas |

No proprietary solver libraries are required. CUDA is used when available.

---

## 2. Algorithm summary

The particle force is factorised into SPGF and Stokes contributions and learned
by five one-input sub-networks with random Fourier features (default `M = 32`):

```
F_total(x, v_x, t) ≈ N_ARFx(x) · N_ARFt(t)
                   + (−c_d N_Sv(v_x))
                   + s · N_Sx(x) · N_St(t)
```

Time integration uses explicit Euler with `dt = 1e-6 s`.

---

## 3. Layout

```
FF-PINN-for-AA/
├── mechanisms/          # Analytical forces + FF-PINN trainers + unified loader
├── initialization/      # Standing-wave acoustics + particle placement
├── PINN/                # Checkpoints (created by training)
├── other_files/         # Simulation drivers (PINN / DEM)
├── scripts/             # Figs. 4–5 force-field plots
├── results/             # Figs. 2–3, 7–13 and Appendix drivers (RFF, Fig7, PD_vs_time, sweeps, kernel)
├── figures/             # Manuscript PNG/PDF outputs
├── validation/          # FEM / COMSOL comparison
├── review/              # This supplementary folder
├── front_page/          # HTML overview
├── FEM_validation.mph   # COMSOL validation project
└── README.md
```

### `mechanisms/`

| File | Role |
|------|------|
| `ARF.py` | Analytical SPGF |
| `Stokes_drag.py` | Analytical Stokes drag |
| `ARF_PINN_x.py` / `ARF_PINN_t.py` | SPGF spatial / temporal networks |
| `STOKES_PINN_x.py` / `STOKES_PINN_t.py` / `STOKES_PINN_v.py` | Stokes factors |
| `UNIFIED_PINN.py` | Combined runtime model |

### `initialization/`

| File | Role |
|------|------|
| `particle_initialization.py` | Particle placement |
| `sound_source_standing.py` | Standing-wave parameters (paper configuration) |

Secondary mechanisms (acoustic wake) and traveling-wave sources are **not** part
of this repository.

---

## 4. Inputs and outputs

- **Inputs:** standing-wave frequency and SPL (`initialization/sound_source_standing.py`),
  particle count / density / diameter, timestep `dt`.
- **Outputs:** particle trajectories, concentration KDE curves, manuscript figures
  under the paths listed in the root `README.md`.

---

## 5. Minimal example

```bash
python review/examples/run_example.py
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full figure map.
