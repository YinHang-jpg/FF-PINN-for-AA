#!/usr/bin/env python3
"""
Compare wake fields on three separate figures: FEM vs PINN, FEM vs DEM, DEM vs PINN.
Particle frame, coordinates x/a, y/a. Figure text is English only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

_COMPARE_DIR = Path(__file__).resolve().parent
_COMSOL_DIR = _COMPARE_DIR.parent
_WAKE_EFFECT_DIR = _COMSOL_DIR.parent
_DEM_WAKE_DIR = _WAKE_EFFECT_DIR / "DEM_wake"
_PINN_WAKE_DIR = _WAKE_EFFECT_DIR / "PINN_wake"
if str(_DEM_WAKE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEM_WAKE_DIR))
if str(_PINN_WAKE_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_WAKE_DIR))

from DEM_wake import (  # noqa: E402
    WAKE_CLOSURE_RE_DEFAULT,
    _A_P,
    acoustic_wake_velocity,
    particle_reynolds_diameter,
)

COLOR_DEM = "#5eb3ff"
COLOR_FEM = "#ff6b6b"
COLOR_PINN = "#4ade80"


def load_comsol_table(path: Path) -> tuple[np.ndarray, float | None]:
    rows: list[list[float]] = []
    detected_time: float | None = None
    time_pattern = re.compile(r"@\s*t\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%"):
                m = time_pattern.search(s)
                if m:
                    detected_time = float(m.group(1))
                continue
            rows.append([float(x) for x in s.split()])
    return np.asarray(rows, dtype=float), detected_time


def build_polar_ring_points(
    radius: float, n_rings: int, center: tuple[float, float] = (0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    if n_rings < 2:
        raise ValueError("n_rings must be >= 2")
    cx, cy = center
    r_samples = np.linspace(radius / n_rings, radius, n_rings)
    x_points: list[float] = [cx]
    y_points: list[float] = [cy]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i, r in enumerate(r_samples, start=1):
        frac = i / n_rings
        n_theta = max(8, int(round(n_rings * (0.6 + 1.8 * frac))))
        theta_offset = (i * golden_angle) % (2.0 * np.pi)
        theta_ring = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + theta_offset
        x_points.extend((cx + r * np.cos(theta_ring)).tolist())
        y_points.extend((cy + r * np.sin(theta_ring)).tolist())
    return np.asarray(x_points, dtype=float), np.asarray(y_points, dtype=float)


def sample_grid(
    domain_radius_m: float, grid: int, particle_radius_m: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_m, y_m = build_polar_ring_points(domain_radius_m, grid)
    outside = np.hypot(x_m, y_m) > particle_radius_m
    x_m = x_m[outside]
    y_m = y_m[outside]
    x_r = x_m / particle_radius_m
    y_r = y_m / particle_radius_m
    return x_m, y_m, x_r, y_r


def dem_velocities(
    x_m: np.ndarray,
    y_m: np.ndarray,
    source_vel: np.ndarray,
    re_closure: float,
) -> tuple[np.ndarray, np.ndarray]:
    src = np.array([0.0, 0.0], dtype=float)
    u = np.zeros_like(x_m)
    v = np.zeros_like(y_m)
    for k in range(x_m.size):
        tgt = np.array([x_m[k], y_m[k]], dtype=float)
        u[k], v[k] = acoustic_wake_velocity(src, tgt, source_vel, Re=re_closure)
    return u, v


def comsol_velocities(
    vx_table: np.ndarray,
    vy_table: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    col = vx_table.shape[1] - 1
    xc = vx_table[:, 0]
    yc = vx_table[:, 1]
    if not np.allclose(vy_table[:, :2], vx_table[:, :2], atol=1e-12):
        raise RuntimeError("COMSOL vx/vy coordinate columns mismatch")
    tri = mtri.Triangulation(xc, yc)
    iu = mtri.LinearTriInterpolator(tri, vx_table[:, col])
    iv = mtri.LinearTriInterpolator(tri, vy_table[:, col])
    u = np.ma.filled(iu(x_m, y_m), 0.0)
    v = np.ma.filled(iv(x_m, y_m), 0.0)
    return u, v


def _norm_quiver(
    u: np.ndarray, v: np.ndarray, arrow_scale_r: float
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-30
    sp = np.hypot(u, v)
    return u / (sp + eps) * arrow_scale_r, v / (sp + eps) * arrow_scale_r


# Quiver often renders shaft linestyle as solid; annotate+FancyArrowPatch respects dashes.
_TOP_DASH_STYLE = (0, (5.0, 4.0))


def _draw_arrows_annotate_dashed(
    ax,
    x_r: np.ndarray,
    y_r: np.ndarray,
    ut: np.ndarray,
    vt: np.ndarray,
    *,
    color: str,
    alpha: float,
    zorder: int,
    lw: float,
    mutation_scale: float,
) -> None:
    for i in range(x_r.size):
        dx, dy = float(ut[i]), float(vt[i])
        if dx * dx + dy * dy < 1e-24:
            continue
        x0, y0 = float(x_r[i]), float(y_r[i])
        ax.annotate(
            "",
            xy=(x0 + dx, y0 + dy),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                linestyle=_TOP_DASH_STYLE,
                color=color,
                alpha=alpha,
                lw=lw,
                mutation_scale=mutation_scale,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=zorder,
        )


def save_pair_figure(
    out_path: Path,
    title: str,
    x_r: np.ndarray,
    y_r: np.ndarray,
    domain_half_r: float,
    arrow_scale_r: float,
    u_bottom: np.ndarray,
    v_bottom: np.ndarray,
    color_bottom: str,
    label_bottom: str,
    u_top: np.ndarray,
    v_top: np.ndarray,
    color_top: str,
    label_top: str,
    width: float,
    footer: str,
    dpi: int,
) -> None:
    """Bottom layer: solid (z=2). Top layer: dashed (z=3)."""
    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ub, vb = _norm_quiver(u_bottom, v_bottom, arrow_scale_r)
    ut, vt = _norm_quiver(u_top, v_top, arrow_scale_r)

    # Shaft length in x/a units; keep within [min_frac, max_frac] of domain half-width
    # so dashed shafts are long enough in pixels for the dash pattern to read.
    kw = dict(
        scale=1,
        scale_units="xy",
        headwidth=4,
        headlength=4,
        headaxislength=3.2,
        minlength=0.05,
    )
    ax.quiver(
        x_r, y_r, ub, vb,
        color=color_bottom, width=width, alpha=0.9, zorder=2,
        linestyle="-",
        **kw,
    )
    fig_w_in = fig.get_figwidth()
    dash_lw = max(0.75, float(width) * fig_w_in * 72.0 * 0.72)
    mut_scale = max(7.0, min(16.0, 4.2 + 0.045 * float(arrow_scale_r)))
    _draw_arrows_annotate_dashed(
        ax, x_r, y_r, ut, vt,
        color=color_top,
        alpha=0.88,
        zorder=3,
        lw=dash_lw,
        mutation_scale=mut_scale,
    )

    particle = Circle(
        (0.0, 0.0), radius=1.0,
        facecolor="#1a2332", edgecolor="white", linewidth=1.2, zorder=5,
    )
    ax.add_patch(particle)
    uk = FancyArrowPatch(
        (0.55, 0.0), (-1.15, 0.0),
        arrowstyle="-|>", mutation_scale=12, color="white", linewidth=1.0, zorder=6,
    )
    ax.add_patch(uk)
    ax.text(-0.35, 0.35, r"$u_k$", color="white", fontsize=12, zorder=6, ha="center", va="bottom")

    ax.set_xlim(-domain_half_r, domain_half_r)
    ax.set_ylim(-domain_half_r, domain_half_r)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x\,/\,a$", color="white", fontsize=11)
    ax.set_ylabel(r"$y\,/\,a$", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_title(title, color="white", fontsize=13, pad=12)

    leg_bottom = Line2D(
        [0], [0], color=color_bottom, lw=2.5, solid_capstyle="round",
        linestyle="-", label=label_bottom,
    )
    leg_top = Line2D(
        [0], [0], color=color_top, lw=2.5, solid_capstyle="round",
        linestyle=_TOP_DASH_STYLE, label=label_top,
    )
    ax.legend(
        handles=[leg_bottom, leg_top],
        loc="upper right",
        facecolor="#161b22",
        edgecolor="#444444",
        labelcolor="white",
        fontsize=10,
    )
    ax.text(0.02, 0.04, footer, transform=ax.transAxes, color="white", fontsize=9, va="bottom")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _print_saved(path: Path) -> None:
    try:
        print(f"Saved: {path}")
    except UnicodeEncodeError:
        print("Saved:", path.as_posix().encode("ascii", errors="replace").decode("ascii"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Save three wake comparison figures (FEM vs PINN, FEM vs DEM, DEM vs PINN)."
    )
    p.add_argument("--grid", type=int, default=18)
    p.add_argument("--domain-radius-m", type=float, default=5e-3)
    p.add_argument("--vx-file", type=Path, default=_COMSOL_DIR / "COMSOL_vx.txt")
    p.add_argument("--vy-file", type=Path, default=_COMSOL_DIR / "COMSOL_vy.txt")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_COMPARE_DIR,
        help="Output directory for PNG files",
    )
    p.add_argument("--particle-vel", type=float, nargs=2, default=[-1.0, 0.0], metavar=("VX", "VY"))
    p.add_argument("--re-closure", type=float, default=WAKE_CLOSURE_RE_DEFAULT)
    p.add_argument("--arrow-scale-m", type=float, default=1.75e-4)
    p.add_argument(
        "--arrow-min-frac",
        type=float,
        default=0.07,
        help="Minimum arrow shaft length as a fraction of domain half-radius (x/a); "
        "raises short arrows so dashed style is visible.",
    )
    p.add_argument(
        "--arrow-max-frac",
        type=float,
        default=0.22,
        help="Cap arrow shaft length as a fraction of domain half-radius (x/a).",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--no-pinn", action="store_true")
    p.add_argument("--pinn-dir", type=Path, default=None)
    args = p.parse_args()

    particle_vel = np.asarray(args.particle_vel, dtype=float)
    u_ref = float(np.linalg.norm(particle_vel))
    re_particle = particle_reynolds_diameter(u_ref)
    domain_half_r = args.domain_radius_m / _A_P
    arrow_scale_r = args.arrow_scale_m / _A_P
    lo = max(0.0, args.arrow_min_frac) * domain_half_r
    hi = max(lo, args.arrow_max_frac * domain_half_r)
    arrow_scale_r = float(np.clip(arrow_scale_r, lo, hi))

    x_m, y_m, x_r, y_r = sample_grid(args.domain_radius_m, args.grid, _A_P)

    vx_tab, t_vx = load_comsol_table(args.vx_file)
    vy_tab, t_vy = load_comsol_table(args.vy_file)
    if vx_tab.shape != vy_tab.shape:
        raise RuntimeError("COMSOL vx/vy shape mismatch")
    if vx_tab.shape[1] < 3:
        raise RuntimeError("COMSOL table needs at least 3 columns")

    u_dem, v_dem = dem_velocities(x_m, y_m, particle_vel, args.re_closure)
    u_com, v_com = comsol_velocities(vx_tab, vy_tab, x_m, y_m)

    u_pinn = v_pinn = None
    if not args.no_pinn:
        try:
            import torch
            from pinn_wake_infer import load_wake_pinn_bundle, pinn_velocities_m_s

            _dev = torch.device("cpu")
            _bundle = load_wake_pinn_bundle(_dev, args.pinn_dir)
            u_pinn, v_pinn = pinn_velocities_m_s(
                x_m, y_m, particle_vel, _A_P, _bundle, _dev
            )
        except Exception as exc:
            print(f"PINN load failed: {exc}", file=sys.stderr)

    final_time = t_vx if t_vx is not None else t_vy
    title_time = "t=last" if final_time is None else f"t={final_time:g} s"
    title_suffix = f"({title_time}, Re={re_particle:.0f}, |U|={u_ref:g} m/s)"
    footer = f"grid={args.grid}, arrows={x_r.size}, Re_closure={args.re_closure:g}"

    out_dir = args.out_dir
    w = 0.0018

    # FEM vs DEM: DEM solid under, FEM dashed on top
    save_pair_figure(
        out_dir / "fem_vs_dem.png",
        f"Wake: FEM vs DEM {title_suffix}",
        x_r, y_r, domain_half_r, arrow_scale_r,
        u_dem, v_dem, COLOR_DEM, "DEM (analytical)",
        u_com, v_com, COLOR_FEM, "FEM (COMSOL)",
        w,
        footer, args.dpi,
    )
    err_fd = np.hypot(u_dem - u_com, v_dem - v_com)
    rmse_fd = float(np.sqrt(np.mean(err_fd * err_fd)))
    cos_fd = float(
        np.mean(
            (u_dem * u_com + v_dem * v_com)
            / (np.hypot(u_dem, v_dem) * np.hypot(u_com, v_com) + 1e-30)
        )
    )

    if u_pinn is not None:
        save_pair_figure(
            out_dir / "fem_vs_pinn.png",
            f"Wake: FEM vs PINN {title_suffix}",
            x_r, y_r, domain_half_r, arrow_scale_r,
            u_pinn, v_pinn, COLOR_PINN, "PINN (surrogate)",
            u_com, v_com, COLOR_FEM, "FEM (COMSOL)",
            w,
            footer, args.dpi,
        )
        save_pair_figure(
            out_dir / "dem_vs_pinn.png",
            f"Wake: DEM vs PINN {title_suffix}",
            x_r, y_r, domain_half_r, arrow_scale_r,
            u_dem, v_dem, COLOR_DEM, "DEM (analytical)",
            u_pinn, v_pinn, COLOR_PINN, "PINN (surrogate)",
            w,
            footer, args.dpi,
        )
        err_pd = np.hypot(u_pinn - u_dem, v_pinn - v_dem)
        rmse_pd = float(np.sqrt(np.mean(err_pd * err_pd)))
        err_fp = np.hypot(u_pinn - u_com, v_pinn - v_com)
        rmse_fp = float(np.sqrt(np.mean(err_fp * err_fp)))
        _print_saved(out_dir / "fem_vs_dem.png")
        _print_saved(out_dir / "fem_vs_pinn.png")
        _print_saved(out_dir / "dem_vs_pinn.png")
        print(f"FEM vs DEM: rmse={rmse_fd:.6f} m/s, mean_cos={cos_fd:.4f}")
        print(f"FEM vs PINN: rmse={rmse_fp:.6f} m/s")
        print(f"DEM vs PINN: rmse={rmse_pd:.6f} m/s")
    else:
        _print_saved(out_dir / "fem_vs_dem.png")
        print(f"FEM vs DEM: rmse={rmse_fd:.6f} m/s, mean_cos={cos_fd:.4f}")
        print("Skipped fem_vs_pinn.png and dem_vs_pinn.png (PINN unavailable)")


if __name__ == "__main__":
    main()
