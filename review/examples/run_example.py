"""
review/examples/run_example.py
==============================

Minimal, self-contained runnable example for the FF-PINN code.

What this example does
----------------------
1. Imports ``run_single_simulation`` from
   ``results/PD_vs_time/PD_time.py`` and integrates the PINN model
   for **0.01 s of physical time** (10 000 steps with dt = 1e-6 s).
2. Reads the corresponding **DEM** baseline column from
   ``results/PD_vs_time/dem_particle_distribution_data.csv`` and the
   **FEM (COMSOL)** reference trajectory from
   ``validation/comsol_positions.csv`` at the closest available
   timestamp.
3. Computes Gaussian-KDE relative-concentration changes for all
   three pipelines and plots a **single-time-point** comparison
   in the same style as
   ``results/PD_vs_time/comparison/density_comparison_group1.png``,
   but with only the ``t = 0.01 s`` curves (PINN red / DEM green /
   FEM blue).

The expected output structure is documented in
``review/examples/expected_output.txt``. Run from the repository
root::

    python review/examples/run_example.py
    python review/examples/run_example.py --plot       # also show the figure interactively

It can also be invoked directly from this folder::

    cd review/examples
    python run_example.py

If the configured PINN checkpoint directory (default: repository ``PINN/``)
does not yet contain **all** factorized weights and JSON normalizers
expected by ``mechanisms/UNIFIED_PINN.py``, this script **runs the five
training modules** in order (``ARF_PINN_x``, ``ARF_PINN_t``, ``STOKES_PINN_x``,
``STOKES_PINN_t``, ``STOKES_PINN_v``) from the repository root before the
demo simulation. That step can take a long time; pass ``--skip-train`` to
fail immediately when checkpoints are missing instead.

Non-default ``pinn_model_base_path`` entries (e.g. ``PINN/freq_10k``) are
**not** populated automatically—the bundled trainers always write under
``PINN/`` at the repo root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_PATH = HERE / "demo_input.json"

# Files required for PD_time / UNIFIED_PINN.load_individual_models (default PINN/).
PINN_ARTIFACT_FILENAMES = (
    "arf_model_x.pth",
    "arf_model_x_normalization_params.json",
    "arf_model_t.pth",
    "arf_model_t_normalization_params.json",
    "stokes_model_x.pth",
    "stokes_model_x_normalization_params.json",
    "stokes_model_t.pth",
    "stokes_model_t_normalization_params.json",
    "stokes_model_v.pth",
    "stokes_model_v_normalization_params.json",
)

TRAINING_SCRIPTS = (
    "mechanisms/ARF_PINN_x.py",
    "mechanisms/ARF_PINN_t.py",
    "mechanisms/STOKES_PINN_x.py",
    "mechanisms/STOKES_PINN_t.py",
    "mechanisms/STOKES_PINN_v.py",
)


def resolve_pinn_model_dir(model_base_path: str | None) -> Path:
    """Absolute directory holding the five sub-network checkpoints."""
    if model_base_path is None:
        return (REPO_ROOT / "PINN").resolve()
    norm = str(model_base_path).strip().replace("\\", "/")
    if not norm or norm in ("PINN", "./PINN"):
        return (REPO_ROOT / "PINN").resolve()
    return (REPO_ROOT / model_base_path).resolve()


def pinn_pack_complete(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in PINN_ARTIFACT_FILENAMES)


def train_default_pinn_pack(*, skip_train: bool) -> None:
    """Ensure ``REPO_ROOT/PINN`` contains all checkpoints; train if missing."""
    model_dir = (REPO_ROOT / "PINN").resolve()
    if pinn_pack_complete(model_dir):
        return
    if skip_train:
        missing = [n for n in PINN_ARTIFACT_FILENAMES if not (model_dir / n).is_file()]
        raise FileNotFoundError(
            f"PINN checkpoints incomplete under {model_dir}. "
            f"Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
            f"Remove --skip-train to run the bundled training scripts, or copy "
            f"a complete weight pack into PINN/."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print("[train] No complete PINN pack found; running factorized training scripts.")
    print("[train] This may take a long time (especially on CPU). Please wait…")
    print("=" * 72 + "\n")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    for rel in TRAINING_SCRIPTS:
        script_path = (REPO_ROOT / rel).resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"Training script not found: {script_path}")
        print(f"[train] >>> {sys.executable} {script_path.relative_to(REPO_ROOT)}")
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            check=True,
        )

    if not pinn_pack_complete(model_dir):
        raise RuntimeError(
            f"Training finished but PINN artifacts are still incomplete under {model_dir}."
        )
    print("\n[train] All PINN checkpoints present; continuing with the example.\n")


def ensure_pinn_ready_for_config(model_base_path: str | None, *, skip_train: bool) -> None:
    """Train default ``PINN/`` when needed; validate custom paths."""
    target = resolve_pinn_model_dir(model_base_path)
    default_root = (REPO_ROOT / "PINN").resolve()

    if pinn_pack_complete(target):
        return

    if target.resolve() == default_root:
        train_default_pinn_pack(skip_train=skip_train)
        return

    missing = [n for n in PINN_ARTIFACT_FILENAMES if not (target / n).is_file()]
    raise FileNotFoundError(
        f"PINN checkpoints incomplete under {target}. Missing (sample): "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
        f"The bundled trainers only write to {default_root}; copy weights into "
        f"'{target.relative_to(REPO_ROOT)}' or set pinn_model_base_path to null."
    )


def load_config(path: Path) -> dict:
    """Load and validate the JSON config used by this example."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    required = [
        "n_steps", "dt_s", "target_time_s",
        "x_window_mm", "kde_sigma_mm", "kde_n_grid",
        "pinn_csv", "dem_csv", "fem_csv",
        "dem_column", "output_figure",
    ]
    # Optional: override checkpoint dir (default = PD_time native ``PINN/``).
    cfg.setdefault("pinn_model_base_path", None)
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return cfg


# ---------------------------------------------------------------------------
# Dynamic import of results/PD_vs_time/PD_time.py
# ---------------------------------------------------------------------------


def import_pd_time_module():
    """Load ``results/PD_vs_time/PD_time.py`` as a module by absolute path."""
    pd_time_path = REPO_ROOT / "results" / "PD_vs_time" / "PD_time.py"
    if not pd_time_path.exists():
        raise FileNotFoundError(
            f"Cannot locate PD_time.py at {pd_time_path}. "
            f"Make sure you run this example from a complete checkout."
        )
    spec = importlib.util.spec_from_file_location("ffpinn_pd_time", pd_time_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # noqa: type: ignore[union-attr]
    return module


# ---------------------------------------------------------------------------
# Density helpers (matching PINN_COMSOL_comparison.py conventions)
# ---------------------------------------------------------------------------


def calculate_particle_density(positions_mm, x_grid_mm, sigma_mm):
    """Gaussian-KDE-style particle number density along x.

    Mirrors ``calculate_particle_density`` in
    ``results/PD_vs_time/comparison/PINN_COMSOL_comparison.py`` so that
    the example reproduces the same normalisation the manuscript uses.
    """
    pos = np.asarray(positions_mm, dtype=float)
    pos = pos[~np.isnan(pos)]
    pos = pos[(pos >= 0.0) & (pos <= 34.0)]
    if pos.size == 0:
        return np.zeros_like(x_grid_mm)
    density = np.zeros_like(x_grid_mm)
    for i, x in enumerate(x_grid_mm):
        d = np.abs(pos - x)
        density[i] = np.sum(np.exp(-(d ** 2) / (2.0 * sigma_mm ** 2)))
    return density


def relative_change(initial_density, final_density):
    """Percentage change relative to the initial density."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = (final_density - initial_density) / initial_density * 100.0
    return np.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# PINN run (delegated to PD_time.run_single_simulation)
# ---------------------------------------------------------------------------


def _patch_model_base_path(pd_time, model_base_path: str | None):
    """Optionally override where checkpoints are loaded from.

    ``PD_time.load_physical_unified_model`` uses ``model_base_path = 'PINN'``
    (files directly under the repository ``PINN/`` folder). That is the
    default for this example when ``pinn_model_base_path`` is omitted,
    null, empty, or the literal ``PINN``.

    Set ``pinn_model_base_path`` in ``demo_input.json`` only if you need a
    non-default directory (e.g. ``PINN/freq_10k``). In that case we replace
    the loader with a thin wrapper that loads from ``REPO_ROOT / path``.
    """
    if model_base_path is None:
        return None
    norm = str(model_base_path).strip().replace("\\", "/")
    if not norm or norm in ("PINN", "./PINN"):
        return None

    from mechanisms.UNIFIED_PINN import (
        load_individual_models, UnifiedNormalizer, PhysicalUnifiedModel,
    )

    abs_path = (REPO_ROOT / model_base_path).resolve()
    if not abs_path.is_dir():
        raise FileNotFoundError(
            f"PINN model base path '{model_base_path}' "
            f"(resolved to {abs_path}) does not exist."
        )

    # Sanity check: ARF-t and Stokes-t JSONs must be trained at the SAME
    # frequency. The PINN/ root pack in this repository accidentally mixes
    # 10 kHz ARF weights with 12 kHz Stokes weights, which puts the two
    # factor branches out of phase and makes the integrator diverge to
    # +/-1e8 mm in a single step. Refuse to run on such a pack rather
    # than silently producing a meaningless density-comparison plot.
    arf_json = abs_path / "arf_model_t_normalization_params.json"
    stk_json = abs_path / "stokes_model_t_normalization_params.json"
    if arf_json.exists() and stk_json.exists():
        with open(arf_json) as fh:
            arf_t_max = float(json.load(fh)["t_max"])
        with open(stk_json) as fh:
            stk_t_max = float(json.load(fh)["t_max"])
        if abs(arf_t_max - stk_t_max) / max(arf_t_max, stk_t_max) > 1e-3:
            raise RuntimeError(
                f"Frequency mismatch in '{model_base_path}': "
                f"arf_model_t was trained with t_max={arf_t_max:.4e} s "
                f"({1/arf_t_max/1000:.2f} kHz) but stokes_model_t was "
                f"trained with t_max={stk_t_max:.4e} s "
                f"({1/stk_t_max/1000:.2f} kHz). "
                f"Replace files under PINN/ with a consistent training "
                f"output, or point pinn_model_base_path at a coherent "
                f"PINN/freq_*k pack."
            )

    original = pd_time.load_physical_unified_model

    def patched(device):
        print(f"[patch] Using PINN weights from {model_base_path}")
        models = load_individual_models(device, str(abs_path))
        unified_norm = UnifiedNormalizer()
        norm_paths = {
            "arf_x_norm":    str(abs_path / "arf_model_x_normalization_params.json"),
            "arf_t_norm":    str(abs_path / "arf_model_t_normalization_params.json"),
            "stokes_x_norm": str(abs_path / "stokes_model_x_normalization_params.json"),
            "stokes_t_norm": str(abs_path / "stokes_model_t_normalization_params.json"),
            "stokes_v_norm": str(abs_path / "stokes_model_v_normalization_params.json"),
        }
        unified_norm.load_from_individual_models(norm_paths, device="cpu")
        return PhysicalUnifiedModel(models, unified_norm, device=device)

    pd_time.load_physical_unified_model = patched
    return original


def run_pinn_simulation(n_steps, position_interval_s, model_base_path=None):
    """Invoke ``PD_time.run_single_simulation`` and return position data."""
    pd_time = import_pd_time_module()
    _patch_model_base_path(pd_time, model_base_path)

    norm = (
        str(model_base_path).strip().replace("\\", "/")
        if model_base_path is not None
        else ""
    )
    if not norm or norm in ("PINN", "./PINN"):
        print("[PINN] Checkpoints: repository root folder PINN/ (PD_time default).")
    else:
        print(f"[PINN] Checkpoints: REPO_ROOT/{norm}")

    print(f"\n[PINN] Running PD_time.run_single_simulation("
          f"total_steps={n_steps}, save_positions=True, "
          f"position_interval={position_interval_s})")

    # Switch the working directory to the repository root so that
    # checkpoints discovered via relative paths inside PD_time.py
    # resolve correctly.
    prev_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        tic = time.perf_counter()
        _, _, position_data = pd_time.run_single_simulation(
            total_steps=n_steps,
            suppress_output=False,
            save_positions=True,
            position_interval=position_interval_s,
        )
        elapsed = time.perf_counter() - tic
    finally:
        os.chdir(prev_cwd)

    print(f"[PINN] Wall time: {elapsed:.2f} s")
    return position_data, elapsed


# ---------------------------------------------------------------------------
# DEM and FEM loaders
# ---------------------------------------------------------------------------


def load_dem_curve(csv_path: Path, column_name: str):
    """Read DEM relative-concentration-change column from the bundled CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"DEM CSV not found at {csv_path}. The bundled file is "
            f"ignored by .gitignore but should be present locally; "
            f"otherwise re-generate it via "
            f"`python results/PD_vs_time/PD_vs_time_DEM.py`."
        )
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    return df["x_mm"].values, df[column_name].values


def load_fem_positions(csv_path: Path, target_time_s: float):
    """Return (x_positions_mm, actual_time, x_initial_mm) at the closest row.

    Mirrors the convention used in
    ``results/PD_vs_time/comparison/PINN_COMSOL_comparison.py``:
    the FEM CSV's first column is time and remaining columns are
    particle x-coordinates.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FEM (COMSOL) CSV not found at {csv_path}. "
            f"This file is ignored by .gitignore but should be present locally."
        )
    df = pd.read_csv(csv_path, header=None)
    times = df.iloc[:, 0].values.astype(float)
    positions = df.iloc[:, 1:].values.astype(float)
    valid = ~np.isnan(times)
    times = times[valid]
    positions = positions[valid, :]

    initial_idx = int(np.argmin(np.abs(times - 0.0)))
    target_idx = int(np.argmin(np.abs(times - target_time_s)))
    return (positions[target_idx, :],
            float(times[target_idx]),
            positions[initial_idx, :])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_single_time_comparison(x_grid_mm, x_window,
                                pinn_rc, dem_rc, fem_rc,
                                pinn_t, fem_t, output_path,
                                show=False):
    """Plot a single-time PINN/DEM/FEM relative-change comparison."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    n_x = len(x_grid_mm)
    markevery = max(1, n_x // 10)
    style = dict(marker="o", markersize=7, markerfacecolor="white",
                 markeredgewidth=1.2, markevery=markevery)

    ax.plot(x_grid_mm, pinn_rc,
            color="red", linestyle="-", linewidth=3.2, alpha=0.9,
            markeredgecolor="red",
            label=f"PINN  t={pinn_t:.2f} s", **style)
    ax.plot(x_grid_mm, dem_rc,
            color="green", linestyle="-", linewidth=2.5, alpha=0.9,
            markeredgecolor="green",
            label="DEM   t=0.01 s", **style)
    ax.plot(x_grid_mm, fem_rc,
            color="blue", linestyle="-", linewidth=1.8, alpha=0.9,
            markeredgecolor="blue",
            label=f"FEM   t={fem_t:.2f} s", **style)

    ax.set_xlim(*x_window)
    ax.set_xlabel("Position x (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative Concentration Change (%)",
                  fontsize=14, fontweight="bold")
    ax.set_title("(a)  Particle density distribution at t = 0.01 s",
                 fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper center", fontsize=11, framealpha=0.95, ncol=3,
              bbox_to_anchor=(0.5, 0.98), handlelength=3.0,
              handletextpad=0.6, labelspacing=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[plot] Saved figure to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FF-PINN minimal example: PD_time.py for t = 0.01 s "
                    "with PINN / DEM / FEM density comparison.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="Path to demo_input.json")
    parser.add_argument("--plot", action="store_true",
                        help="Show the matplotlib figure interactively in "
                             "addition to writing the PNG file.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Do not run bundled training scripts when PINN/ "
                             "is incomplete; exit with an error instead.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_pinn_ready_for_config(
        cfg.get("pinn_model_base_path"),
        skip_train=args.skip_train,
    )

    print("=" * 72)
    print("FF-PINN review/examples/run_example.py")
    print("=" * 72)
    print(f"Config              : {args.config}")
    print(f"Repository root     : {REPO_ROOT}")
    print(f"Steps / dt          : {cfg['n_steps']} steps x {cfg['dt_s']:.0e} s "
          f"= {cfg['n_steps'] * cfg['dt_s']:.4f} s simulated")

    # ------------------------------------------------------------------
    # 1. PINN simulation via PD_time.run_single_simulation
    # ------------------------------------------------------------------
    pinn_data, pinn_wall_time = run_pinn_simulation(
        n_steps=cfg["n_steps"],
        position_interval_s=cfg["target_time_s"],
        model_base_path=cfg.get("pinn_model_base_path"),
    )
    pinn_times = np.asarray(pinn_data["times"])
    pinn_positions = pinn_data["positions"]

    if len(pinn_times) < 2:
        raise RuntimeError(
            f"Expected at least two snapshots (t=0 and t={cfg['target_time_s']} s) "
            f"from PD_time.run_single_simulation, got {len(pinn_times)}."
        )
    pinn_initial_pos_mm = np.asarray(pinn_positions[0])
    pinn_target_idx = int(np.argmin(np.abs(pinn_times - cfg["target_time_s"])))
    pinn_target_t = float(pinn_times[pinn_target_idx])
    pinn_final_pos_mm = np.asarray(pinn_positions[pinn_target_idx])

    print(f"[PINN] Snapshots at t = {pinn_times.tolist()}")
    print(f"[PINN] Using snapshot closest to {cfg['target_time_s']} s "
          f"-> t = {pinn_target_t:.4f} s ({pinn_final_pos_mm.size} particles)")

    # ------------------------------------------------------------------
    # 2. KDE grid and reference initial density (initial uniform layout)
    # ------------------------------------------------------------------
    x_grid_full_mm = np.linspace(0.0, 34.0, cfg["kde_n_grid"])
    x_window = tuple(cfg["x_window_mm"])
    x_mask = (x_grid_full_mm >= x_window[0]) & (x_grid_full_mm <= x_window[1])
    x_grid_mm = x_grid_full_mm[x_mask]
    sigma = float(cfg["kde_sigma_mm"])

    pinn_init_density = calculate_particle_density(
        pinn_initial_pos_mm, x_grid_full_mm, sigma)
    pinn_final_density = calculate_particle_density(
        pinn_final_pos_mm, x_grid_full_mm, sigma)
    pinn_rc = relative_change(pinn_init_density, pinn_final_density)[x_mask]

    print(f"[PINN] Relative change range "
          f"[{pinn_rc.min():+.2f} %, {pinn_rc.max():+.2f} %]")

    # ------------------------------------------------------------------
    # 3. DEM curve (pre-computed bundled CSV)
    # ------------------------------------------------------------------
    dem_csv = REPO_ROOT / cfg["dem_csv"]
    print(f"\n[DEM ] Loading {dem_csv}")
    x_dem_mm, dem_rc_full = load_dem_curve(dem_csv, cfg["dem_column"])

    # The DEM CSV mirrors the spatial axis around x = 17 mm in the
    # original comparison script; replicate that flip exactly so the
    # plot matches density_comparison_group1.png (a).
    center_x = 0.5 * (x_window[0] + x_window[1])
    x_dem_flipped = 2.0 * center_x - x_dem_mm
    from scipy.interpolate import interp1d
    dem_interp = interp1d(x_dem_flipped, dem_rc_full, kind="linear",
                          bounds_error=False, fill_value="extrapolate")
    dem_rc = dem_interp(x_grid_mm)
    print(f"[DEM ] Relative change range "
          f"[{dem_rc.min():+.2f} %, {dem_rc.max():+.2f} %]")

    # ------------------------------------------------------------------
    # 4. FEM (COMSOL) curve
    # ------------------------------------------------------------------
    fem_csv = REPO_ROOT / cfg["fem_csv"]
    print(f"\n[FEM ] Loading {fem_csv}")
    fem_pos_mm, fem_t, fem_initial_mm = load_fem_positions(
        fem_csv, cfg["target_time_s"])
    print(f"[FEM ] Snapshot closest to {cfg['target_time_s']} s "
          f"-> t = {fem_t:.4f} s ({fem_pos_mm.size} particles)")

    fem_init_density = calculate_particle_density(
        fem_initial_mm, x_grid_full_mm, sigma)
    fem_final_density = calculate_particle_density(
        fem_pos_mm, x_grid_full_mm, sigma)
    fem_rc = relative_change(fem_init_density, fem_final_density)[x_mask]
    print(f"[FEM ] Relative change range "
          f"[{fem_rc.min():+.2f} %, {fem_rc.max():+.2f} %]")

    # ------------------------------------------------------------------
    # 5. Plot single-time-point comparison
    # ------------------------------------------------------------------
    output_fig = HERE / cfg["output_figure"]
    plot_single_time_comparison(
        x_grid_mm=x_grid_mm,
        x_window=x_window,
        pinn_rc=pinn_rc,
        dem_rc=dem_rc,
        fem_rc=fem_rc,
        pinn_t=pinn_target_t,
        fem_t=fem_t,
        output_path=output_fig,
        show=args.plot,
    )

    # ------------------------------------------------------------------
    # 6. Summary block
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Summary")
    print("-" * 72)
    print(f"  PINN target t       : {pinn_target_t:.4f} s "
          f"(wall time {pinn_wall_time:.2f} s)")
    print(f"  PINN range          : "
          f"[{pinn_rc.min():+.2f} %, {pinn_rc.max():+.2f} %]")
    print(f"  DEM  range          : "
          f"[{dem_rc.min():+.2f} %, {dem_rc.max():+.2f} %]")
    print(f"  FEM  target t       : {fem_t:.4f} s")
    print(f"  FEM  range          : "
          f"[{fem_rc.min():+.2f} %, {fem_rc.max():+.2f} %]")
    rmse = float(np.sqrt(np.mean((pinn_rc - fem_rc) ** 2)))
    mae = float(np.mean(np.abs(pinn_rc - fem_rc)))
    print(f"  PINN vs. FEM RMSE   : {rmse:.2f} %")
    print(f"  PINN vs. FEM MAE    : {mae:.2f} %")
    print(f"  Output figure       : {output_fig.relative_to(REPO_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
