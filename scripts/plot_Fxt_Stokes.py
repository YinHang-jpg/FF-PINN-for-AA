"""Manuscript Fig. 5 — Stokes drag F(x, v_rel, t) from trained PINN spatial/temporal factors."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "results"))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from _figures import FIGURES  # noqa: E402

frequency = 10000  # Hz
sound_speed = 340  # m/s
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083

FS_LABEL = 16
FS_TICK = 14
FS_SUBTITLE = 15
FS_CBAR_LABEL = 15
FS_CBAR_TICK = 13
OUT_PATH = FIGURES / "fig05_stokes_force_3d_visualization.png"


def load_stokes_xt_models():
    from mechanisms.STOKES_PINN_x import StokesNetX
    from mechanisms.STOKES_PINN_t import StokesNetT

    state_x = torch.load(_ROOT / "PINN" / "stokes_model_x.pth", map_location="cpu")
    model_x = StokesNetX(fourier_features=32)
    model_x.load_state_dict(state_x, strict=True)
    model_x.eval()
    with open(_ROOT / "PINN" / "stokes_model_x_normalization_params.json", "r", encoding="utf-8") as f:
        norms_x = json.load(f)

    state_t = torch.load(_ROOT / "PINN" / "stokes_model_t.pth", map_location="cpu")
    period_s = 1.0 / frequency
    model_t = StokesNetT(period_seconds=period_s)
    model_t.load_state_dict(state_t, strict=True)
    model_t.eval()
    with open(_ROOT / "PINN" / "stokes_model_t_normalization_params.json", "r", encoding="utf-8") as f:
        norms_t = json.load(f)

    return {"model_x": model_x, "norms_x": norms_x,
            "model_t": model_t, "norms_t": norms_t}


def predict_stokes_factor(model, norms, values, key_prefix):
    with torch.no_grad():
        tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        if "x" in key_prefix:
            val_min, val_max = norms["x_min"], norms["x_max"]
        else:
            val_min, val_max = norms["t_min"], norms["t_max"]
        scaled = (tensor - val_min) / (val_max - val_min)
        pred_norm = model(scaled)
        return (pred_norm * float(norms["force_sigma"]) + float(norms["force_mu"])
                ).squeeze(1).cpu().numpy().astype(np.float64)


def stokes_drag_coeff():
    return 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham


def stokes_force_pN(fx, ft, v_rel_ms, b):
    """Factorized Stokes drag for visualization [pN]: F = -b * v_rel * Fx * Ft."""
    return -b * np.asarray(v_rel_ms) * np.asarray(fx) * np.asarray(ft) * 1e12


def _smooth_heatmap(ax, x_1d, y_1d, Z_pN):
    zmax = float(np.nanmax(np.abs(Z_pN)))
    if zmax < 1e-30:
        zmax = 1.0
    norm = TwoSlopeNorm(vmin=-zmax, vcenter=0.0, vmax=zmax)
    dx = (x_1d[1] - x_1d[0]) if len(x_1d) > 1 else 1.0
    dy = (y_1d[1] - y_1d[0]) if len(y_1d) > 1 else 1.0
    x_edges = np.concatenate([x_1d - 0.5 * dx, [x_1d[-1] + 0.5 * dx]])
    y_edges = np.concatenate([y_1d - 0.5 * dy, [y_1d[-1] + 0.5 * dy]])
    X_e, Y_e = np.meshgrid(x_edges, y_edges)
    mesh = ax.pcolormesh(X_e, Y_e, Z_pN, cmap="RdBu_r", norm=norm,
                         shading="flat", rasterized=True)
    ax.set_xlim(x_1d[0], x_1d[-1])
    ax.set_ylim(y_1d[0], y_1d[-1])
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("F (pN)", fontsize=FS_CBAR_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    return mesh


def main():
    print("Loading Stokes Fx/Ft PINN models...")
    models = load_stokes_xt_models()
    b = stokes_drag_coeff()
    print(f"drag_coeff b = {b:.3e} N/(m/s)")

    wavelength = sound_speed / frequency
    x_range = 3 * wavelength / 2
    x_min_m, x_max_m = -x_range / 2, x_range / 2
    t_min_s, t_max_s = 0.0, 1.0 / frequency
    v_rel_min, v_rel_max = -0.1, 0.1

    num_points = 320
    xs_m = np.linspace(x_min_m, x_max_m, num_points)
    v_rel_ms = np.linspace(v_rel_min, v_rel_max, num_points)
    ts_s = np.linspace(t_min_s, t_max_s, num_points)

    print("Predicting Fx(x), Ft(t)...")
    F_x = predict_stokes_factor(models["model_x"], models["norms_x"], xs_m, "x")
    F_t = predict_stokes_factor(models["model_t"], models["norms_t"], ts_s, "t")

    X_mm = xs_m * 1000.0
    V_rel_mms = v_rel_ms * 1000.0
    T_us = ts_s * 1e6

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    plt.subplots_adjust(wspace=0.38, left=0.05, right=0.97, top=0.88, bottom=0.14)

    v_idx = int(num_points * 0.75)
    v_rel_fixed = float(v_rel_ms[v_idx])
    Fx_xt, Ft_xt = np.meshgrid(F_x, F_t)
    F_xt = -stokes_force_pN(Fx_xt, Ft_xt, v_rel_fixed, b)
    ax = axes[0]
    _smooth_heatmap(ax, X_mm, T_us, F_xt)
    ax.set_xlabel("Position x (mm)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("Time t (μs)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"F(x,t) at $v_{{\\mathrm{{rel}}}}$ = {v_rel_fixed * 1000:.1f} mm/s",
        fontsize=FS_SUBTITLE, fontweight="bold", pad=12,
    )
    ax.tick_params(labelsize=FS_TICK, width=1.2, length=5)

    t_idx = num_points // 4
    ft_fixed = float(F_t[t_idx])
    F_xv = -stokes_force_pN(F_x[None, :], ft_fixed, v_rel_ms[:, None], b)
    ax = axes[1]
    _smooth_heatmap(ax, X_mm, V_rel_mms, F_xv)
    ax.set_xlabel("Position x (mm)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("Relative velocity $v_{\\mathrm{rel}}$ (mm/s)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"F(x, $v_{{\\mathrm{{rel}}}}$) at t = {ts_s[t_idx] * 1e6:.1f} μs",
        fontsize=FS_SUBTITLE, fontweight="bold", pad=12,
    )
    ax.tick_params(labelsize=FS_TICK, width=1.2, length=5)

    x_idx = int(num_points * 0.75)
    fx_fixed = float(F_x[x_idx])
    F_vt = -stokes_force_pN(fx_fixed, F_t[:, None], v_rel_ms[None, :], b)
    ax = axes[2]
    _smooth_heatmap(ax, V_rel_mms, T_us, F_vt)
    ax.set_xlabel("Relative velocity $v_{\\mathrm{rel}}$ (mm/s)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_ylabel("Time t (μs)", fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(
        f"F($v_{{\\mathrm{{rel}}}}$, t) at x = {xs_m[x_idx] * 1000:.1f} mm",
        fontsize=FS_SUBTITLE, fontweight="bold", pad=12,
    )
    ax.tick_params(labelsize=FS_TICK, width=1.2, length=5)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=400, bbox_inches="tight", facecolor="white")
    print("Saved:", OUT_PATH, "bytes=", OUT_PATH.stat().st_size)
    print(f"F(x,t) range [pN]: [{F_xt.min():.2f}, {F_xt.max():.2f}]")
    print(f"F(x,v) range [pN]: [{F_xv.min():.2f}, {F_xv.max():.2f}]")
    print(f"F(v,t) range [pN]: [{F_vt.min():.2f}, {F_vt.max():.2f}]")
    plt.close(fig)


if __name__ == "__main__":
    main()
