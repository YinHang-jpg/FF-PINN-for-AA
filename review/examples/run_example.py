"""
review/examples/run_example.py — FF-PINN minimal demo (PINN vs DEM vs FEM at t ≈ 0.01 s).

Runs ``results/PD_vs_time/PD_time.run_single_simulation``, compares KDE relative
concentration change to DEM CSV + COMSOL FEM, saves ``demo_input.json`` figure.

Usage (repo root)::

    python review/examples/run_example.py
    python review/examples/run_example.py --plot
    python review/examples/run_example.py --skip-train   # fail if PINN/ incomplete

If ``PINN/`` lacks all factor checkpoints, runs the five ``mechanisms/*_PINN_*.py``
trainers (slow). Trainers only populate ``PINN/``; custom ``pinn_model_base_path``
dirs must be filled manually. See ``review/examples/expected_output.txt``.
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
from scipy.interpolate import interp1d

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.resolve()
PINN_DIR = (REPO_ROOT / "PINN").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CONFIG = HERE / "demo_input.json"

PINN_ARTIFACTS = (
    "arf_model_x.pth", "arf_model_x_normalization_params.json",
    "arf_model_t.pth", "arf_model_t_normalization_params.json",
    "stokes_model_x.pth", "stokes_model_x_normalization_params.json",
    "stokes_model_t.pth", "stokes_model_t_normalization_params.json",
    "stokes_model_v.pth", "stokes_model_v_normalization_params.json",
)

TRAIN_SCRIPTS = (
    "mechanisms/ARF_PINN_x.py", "mechanisms/ARF_PINN_t.py",
    "mechanisms/STOKES_PINN_x.py", "mechanisms/STOKES_PINN_t.py",
    "mechanisms/STOKES_PINN_v.py",
)


def checkpoint_dir(model_base_path: str | None) -> Path:
    """Directory holding the five sub-networks (default: ``PINN/``)."""
    if model_base_path is None:
        return PINN_DIR
    s = str(model_base_path).strip().replace("\\", "/")
    if not s or s in ("PINN", "./PINN"):
        return PINN_DIR
    return (REPO_ROOT / model_base_path).resolve()


def _pack_complete(d: Path) -> bool:
    return all((d / n).is_file() for n in PINN_ARTIFACTS)


def _train_if_needed(skip_train: bool) -> None:
    if _pack_complete(PINN_DIR):
        return
    if skip_train:
        miss = [n for n in PINN_ARTIFACTS if not (PINN_DIR / n).is_file()]
        raise FileNotFoundError(
            f"Incomplete PINN pack under {PINN_DIR}; missing e.g. {miss[:3]}… "
            f"Omit --skip-train to train, or copy weights into PINN/."
        )
    PINN_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[train] PINN/ incomplete — running factorized trainers (may take a long time).\n")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    root = str(REPO_ROOT)
    prev = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = root + ((os.pathsep + prev) if prev else "")

    for rel in TRAIN_SCRIPTS:
        script = REPO_ROOT / rel
        if not script.is_file():
            raise FileNotFoundError(script)
        print(f"[train] {sys.executable} {rel}")
        subprocess.run([sys.executable, str(script)], cwd=root, env=env, check=True)

    if not _pack_complete(PINN_DIR):
        raise RuntimeError(f"Training finished but PINN/ still incomplete: {PINN_DIR}")


def ensure_weights(cfg_path: str | None, *, skip_train: bool) -> None:
    target = checkpoint_dir(cfg_path)
    if _pack_complete(target):
        return
    if target != PINN_DIR:
        miss = [n for n in PINN_ARTIFACTS if not (target / n).is_file()]
        raise FileNotFoundError(
            f"Incomplete checkpoints under {target} (e.g. missing {miss[:3]}…). "
            f"Bundled trainers only write to PINN/ — copy files into "
            f"{target.relative_to(REPO_ROOT)} or set pinn_model_base_path to null."
        )
    _train_if_needed(skip_train)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    req = (
        "n_steps", "dt_s", "target_time_s", "x_window_mm", "kde_sigma_mm",
        "kde_n_grid", "pinn_csv", "dem_csv", "fem_csv", "dem_column", "output_figure",
    )
    cfg.setdefault("pinn_model_base_path", None)
    bad = [k for k in req if k not in cfg]
    if bad:
        raise KeyError(f"Missing keys in {path}: {bad}")
    return cfg


def import_pd_time():
    path = REPO_ROOT / "results" / "PD_vs_time" / "PD_time.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ffpinn_pd_time", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def calculate_particle_density(positions_mm, x_grid_mm, sigma_mm):
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
    with np.errstate(divide="ignore", invalid="ignore"):
        rc = (final_density - initial_density) / initial_density * 100.0
    return np.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0)


def _patch_loader(pd_time, model_base_path: str | None) -> None:
    """Patch PD_time to load from a non-default directory under REPO_ROOT."""
    root = checkpoint_dir(model_base_path)
    if root == PINN_DIR:
        return

    from mechanisms.UNIFIED_PINN import (
        PhysicalUnifiedModel, UnifiedNormalizer, load_individual_models,
    )

    if not root.is_dir():
        raise FileNotFoundError(f"PINN base path does not exist: {root}")

    arf_j, stk_j = root / "arf_model_t_normalization_params.json", root / "stokes_model_t_normalization_params.json"
    if arf_j.is_file() and stk_j.is_file():
        with open(arf_j, encoding="utf-8") as fa, open(stk_j, encoding="utf-8") as fs:
            arf_tm = float(json.load(fa)["t_max"])
            stk_tm = float(json.load(fs)["t_max"])
        if abs(arf_tm - stk_tm) / max(arf_tm, stk_tm) > 1e-3:
            raise RuntimeError(
                f"ARF-t vs Stokes-t frequency mismatch under {model_base_path}: "
                f"t_max {arf_tm:.4e} vs {stk_tm:.4e} s — use a coherent PINN pack."
            )

    def patched(device):
        print(f"[PINN] Loading weights from {model_base_path}")
        models = load_individual_models(device, str(root))
        unified_norm = UnifiedNormalizer()
        unified_norm.load_from_individual_models(
            {
                "arf_x_norm": str(root / "arf_model_x_normalization_params.json"),
                "arf_t_norm": str(root / "arf_model_t_normalization_params.json"),
                "stokes_x_norm": str(root / "stokes_model_x_normalization_params.json"),
                "stokes_t_norm": str(root / "stokes_model_t_normalization_params.json"),
                "stokes_v_norm": str(root / "stokes_model_v_normalization_params.json"),
            },
            device="cpu",
        )
        return PhysicalUnifiedModel(models, unified_norm, device=device)

    pd_time.load_physical_unified_model = patched


def run_pinn_simulation(n_steps, position_interval_s, model_base_path=None):
    pd_time = import_pd_time()
    _patch_loader(pd_time, model_base_path)
    root = checkpoint_dir(model_base_path)
    label = "PINN/ (default)" if root == PINN_DIR else str(root.relative_to(REPO_ROOT))
    print(f"[PINN] Checkpoints: {label}")
    print(f"[PINN] run_single_simulation(steps={n_steps}, interval={position_interval_s})")

    prev = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        t0 = time.perf_counter()
        _, _, position_data = pd_time.run_single_simulation(
            total_steps=n_steps,
            suppress_output=False,
            save_positions=True,
            position_interval=position_interval_s,
        )
        elapsed = time.perf_counter() - t0
    finally:
        os.chdir(prev)

    print(f"[PINN] Wall time: {elapsed:.2f} s")
    return position_data, elapsed


def load_dem_curve(csv_path: Path, column_name: str):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} — regenerate with results/PD_vs_time/PD_vs_time_DEM.py if needed."
        )
    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise KeyError(f"No column '{column_name}' in {csv_path}: {list(df.columns)}")
    return df["x_mm"].values, df[column_name].values


def load_fem_positions(csv_path: Path, target_time_s: float):
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, header=None)
    times = df.iloc[:, 0].values.astype(float)
    positions = df.iloc[:, 1:].values.astype(float)
    valid = ~np.isnan(times)
    times, positions = times[valid], positions[valid, :]
    i0 = int(np.argmin(np.abs(times - 0.0)))
    i1 = int(np.argmin(np.abs(times - target_time_s)))
    return positions[i1, :], float(times[i1]), positions[i0, :]


def plot_comparison(x_grid_mm, x_window, pinn_rc, dem_rc, fem_rc,
                    pinn_t, fem_t, output_path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    n_x = len(x_grid_mm)
    me = max(1, n_x // 10)
    sty = dict(marker="o", markersize=7, markerfacecolor="white",
               markeredgewidth=1.2, markevery=me)

    ax.plot(x_grid_mm, pinn_rc, color="red", lw=3.2, alpha=0.9,
            markeredgecolor="red", label=f"PINN  t={pinn_t:.2f} s", **sty)
    ax.plot(x_grid_mm, dem_rc, color="green", lw=2.5, alpha=0.9,
            markeredgecolor="green", label="DEM   t=0.01 s", **sty)
    ax.plot(x_grid_mm, fem_rc, color="blue", lw=1.8, alpha=0.9,
            markeredgecolor="blue", label=f"FEM   t={fem_t:.2f} s", **sty)

    ax.set_xlim(*x_window)
    ax.set_xlabel("Position x (mm)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Relative Concentration Change (%)", fontsize=14, fontweight="bold")
    ax.set_title("(a)  Particle density distribution at t = 0.01 s", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.axhline(y=0, color="black", lw=0.8, alpha=0.5)
    ax.legend(loc="upper center", fontsize=11, framealpha=0.95, ncol=3,
              bbox_to_anchor=(0.5, 0.98), handlelength=3.0,
              handletextpad=0.6, labelspacing=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[plot] {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="FF-PINN density demo (PD_time + DEM + FEM).")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config path")
    ap.add_argument("--plot", action="store_true", help="Show matplotlib window")
    ap.add_argument("--skip-train", action="store_true",
                  help="Abort if PINN/ incomplete instead of running trainers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_weights(cfg.get("pinn_model_base_path"), skip_train=args.skip_train)

    print("=" * 72)
    print(f"run_example.py | root={REPO_ROOT}")
    print(f"config={args.config} | simulated time = {cfg['n_steps'] * cfg['dt_s']:.4f} s")
    print("=" * 72)

    pinn_data, wall = run_pinn_simulation(
        cfg["n_steps"], cfg["target_time_s"], cfg.get("pinn_model_base_path"),
    )
    pt = np.asarray(pinn_data["times"])
    pp = pinn_data["positions"]
    if len(pt) < 2:
        raise RuntimeError(f"Need ≥2 PINN snapshots, got {len(pt)}.")

    pinn_i = np.asarray(pp[0])
    j = int(np.argmin(np.abs(pt - cfg["target_time_s"])))
    pinn_t, pinn_f = float(pt[j]), np.asarray(pp[j])
    print(f"[PINN] t snapshots {pt.tolist()} → using t={pinn_t:.4f} s, N={pinn_f.size}")

    x_full = np.linspace(0.0, 34.0, cfg["kde_n_grid"])
    x_win = tuple(cfg["x_window_mm"])
    mask = (x_full >= x_win[0]) & (x_full <= x_win[1])
    xg = x_full[mask]
    sig = float(cfg["kde_sigma_mm"])

    pi = calculate_particle_density(pinn_i, x_full, sig)
    pf = calculate_particle_density(pinn_f, x_full, sig)
    pinn_rc = relative_change(pi, pf)[mask]
    print(f"[PINN] relative change [%]: [{pinn_rc.min():+.2f}, {pinn_rc.max():+.2f}]")

    dem_p = REPO_ROOT / cfg["dem_csv"]
    print(f"[DEM] {dem_p}")
    xd, dem_full = load_dem_curve(dem_p, cfg["dem_column"])
    cx = 0.5 * (x_win[0] + x_win[1])
    dem_rc = interp1d(2.0 * cx - xd, dem_full, kind="linear",
                      bounds_error=False, fill_value="extrapolate")(xg)
    print(f"[DEM] relative change [%]: [{dem_rc.min():+.2f}, {dem_rc.max():+.2f}]")

    fem_p = REPO_ROOT / cfg["fem_csv"]
    print(f"[FEM] {fem_p}")
    fem_pos, fem_t, fem_i = load_fem_positions(fem_p, cfg["target_time_s"])
    print(f"[FEM] t≈{cfg['target_time_s']} s → {fem_t:.4f} s, N={fem_pos.size}")
    fi = calculate_particle_density(fem_i, x_full, sig)
    ff = calculate_particle_density(fem_pos, x_full, sig)
    fem_rc = relative_change(fi, ff)[mask]
    print(f"[FEM] relative change [%]: [{fem_rc.min():+.2f}, {fem_rc.max():+.2f}]")

    out = HERE / cfg["output_figure"]
    plot_comparison(xg, x_win, pinn_rc, dem_rc, fem_rc, pinn_t, fem_t, out, args.plot)

    print("-" * 72)
    rmse = float(np.sqrt(np.mean((pinn_rc - fem_rc) ** 2)))
    mae = float(np.mean(np.abs(pinn_rc - fem_rc)))
    print(f"PINN vs FEM RMSE {rmse:.2f} % | MAE {mae:.2f} %")
    print(f"Figure: {out.relative_to(REPO_ROOT)} | PINN wall {wall:.2f} s")
    print("Done.")


if __name__ == "__main__":
    main()
