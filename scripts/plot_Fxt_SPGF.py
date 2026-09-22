"""Manuscript Fig. 4 — SPGF / ARF force field F(x,t) from trained PINN factors."""
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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from _figures import FIGURES  # noqa: E402

try:
    from mechanisms.ARF import c_0, frequency
except Exception:
    from ARF import c_0, frequency

FS_LABEL = 16
FS_TICK = 14
FS_SUBTITLE = 15
FS_CBAR_LABEL = 15
FS_CBAR_TICK = 13
OUT_PATH = FIGURES / "fig04_spgf_force_visualization.png"


def load_x_model_and_norms():
    from mechanisms.ARF_PINN_x import ARFNet as TrainedARFNetX

    state = torch.load(_ROOT / "PINN" / "arf_model_x.pth", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = TrainedARFNetX(fourier_features=32)
    model.load_state_dict(state, strict=True)
    model.eval()

    with open(_ROOT / "PINN" / "arf_model_x_normalization_params.json", "r", encoding="utf-8") as f:
        norms = json.load(f)
    return model, norms


def load_t_model_and_norms():
    from mechanisms.ARF_PINN_t import ARFNetT as TrainedARFNetT

    state = torch.load(_ROOT / "PINN" / "arf_model_t.pth", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    with open(_ROOT / "PINN" / "arf_model_t_normalization_params.json", "r", encoding="utf-8") as f:
        norms = json.load(f)

    model = TrainedARFNetT(period_seconds=1.0 / frequency)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, norms


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
    mesh = ax.pcolormesh(
        X_e, Y_e, Z_pN,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    ax.set_xlim(x_1d[0], x_1d[-1])
    ax.set_ylim(y_1d[0], y_1d[-1])
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("F (pN)", fontsize=FS_CBAR_LABEL, fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_CBAR_TICK)
    return mesh


def main():
    print("Loading ARF / SPGF PINN models...")
    model_x, norms_x = load_x_model_and_norms()
    model_t, norms_t = load_t_model_and_norms()

    x_min_m = float(norms_x["x_min"])
    x_max_m = float(norms_x["x_max"])
    t_min_s = float(norms_t["t_min"])
    t_max_s = float(norms_t["t_max"])

    num_x = 320
    num_t = 320
    xs_m = np.linspace(x_min_m, x_max_m, num_x, dtype=np.float64)
    ts_s = np.linspace(t_min_s, t_max_s, num_t, dtype=np.float64)

    with torch.no_grad():
        x_tensor = torch.tensor(xs_m, dtype=torch.float32).unsqueeze(1)
        x_scaled = (x_tensor - x_min_m) / (x_max_m - x_min_m)
        pred_x_norm = model_x(x_scaled)
        F_x = (pred_x_norm * float(norms_x["force_sigma"]) + float(norms_x["force_mu"])
               ).squeeze(1).cpu().numpy().astype(np.float64)

        t_tensor = torch.tensor(ts_s, dtype=torch.float32).unsqueeze(1)
        t_norm = (t_tensor - t_min_s) / (t_max_s - t_min_s)
        pred_t_norm = model_t(t_norm)
        T_t = (pred_t_norm * float(norms_t["time_factor_sigma"]) + float(norms_t["time_factor_mu"])
               ).squeeze(1).cpu().numpy().astype(np.float64)

    F_xt = np.outer(T_t, F_x)
    F_xt_pN = F_xt * 1e12

    X_mm = xs_m * 1000.0
    T_us = ts_s * 1e6
    X_grid, T_grid = np.meshgrid(X_mm, T_us)

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        left=0.06, right=0.94, top=0.90, bottom=0.14, wspace=0.30,
    )

    ax3d = fig.add_subplot(gs[0], projection="3d")
    surf = ax3d.plot_surface(
        X_grid, T_grid, F_xt_pN,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=2,
        cstride=2,
        shade=True,
    )
    ax3d.set_xlabel("x (mm)", fontsize=FS_LABEL, fontweight="bold", labelpad=12)
    ax3d.set_ylabel("t (μs)", fontsize=FS_LABEL, fontweight="bold", labelpad=12)
    ax3d.set_zlabel("F (pN)", fontsize=FS_LABEL, fontweight="bold", labelpad=10)
    ax3d.set_title("PINN-predicted F(x,t)", fontsize=FS_SUBTITLE, fontweight="bold", pad=14)
    ax3d.tick_params(labelsize=FS_TICK, pad=4)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.view_init(elev=28, azim=-135)
    cbar3d = fig.colorbar(surf, ax=ax3d, shrink=0.65, pad=0.10, fraction=0.035)
    cbar3d.set_label("F (pN)", fontsize=FS_CBAR_LABEL, fontweight="bold")
    cbar3d.ax.tick_params(labelsize=FS_CBAR_TICK)

    ax2 = fig.add_subplot(gs[1])
    _smooth_heatmap(ax2, X_mm, T_us, F_xt_pN)
    ax2.set_xlabel("x (mm)", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_ylabel("t (μs)", fontsize=FS_LABEL, fontweight="bold")
    ax2.set_title("PINN-predicted F(x,t) contour", fontsize=FS_SUBTITLE, fontweight="bold", pad=14)
    ax2.tick_params(labelsize=FS_TICK, width=1.2, length=5)
    ax2.set_aspect("auto")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=400, bbox_inches="tight", facecolor="white")
    print("Saved:", OUT_PATH, "bytes=", OUT_PATH.stat().st_size)
    print(f"F(x,t) range [pN]: [{F_xt_pN.min():.2f}, {F_xt_pN.max():.2f}]")
    plt.close(fig)


if __name__ == "__main__":
    main()
