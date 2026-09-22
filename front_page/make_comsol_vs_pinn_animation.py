"""Build 10 kHz FF-PINN vs FEM (COMSOL) GIF for t = 0–1 ms."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from initialization.sound_source_standing import sound_pressure_level, spl_to_pressure

ANIM_FREQUENCY_HZ = 10000.0
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "anim_data"
DEFAULT_FEM_CSV = PROJECT_ROOT / "results" / "PD_vs_time" / "fem_particle_positions.csv"
DEFAULT_T_MAX_S = 0.001
DEFAULT_VIDEO_SECONDS = 12.0
CURVE_SMOOTH_SIGMA = 1.5

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 15
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12


def concentration(x_mm, x_grid_mm, sigma=0.25):
    x_mm = np.asarray(x_mm, dtype=np.float64)
    x_mm = x_mm[np.isfinite(x_mm)]
    if x_mm.size == 0:
        return np.zeros_like(x_grid_mm)
    d = x_grid_mm[:, None] - x_mm[None, :]
    return np.sum(np.exp(-(d * d) / (2.0 * sigma * sigma)), axis=1)


def relative_change(x_mm, c0, x_grid_mm, smooth_sigma=0.0):
    c = concentration(x_mm, x_grid_mm)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (c - c0) / c0 * 100.0
    rel = np.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma > 0.0:
        rel = gaussian_filter1d(rel, sigma=float(smooth_sigma), mode="nearest")
    return rel


def cosine_sound_field(domain_size=(0.034, 0.034), resolution=(220, 220)):
    lx, ly = domain_size
    nx, ny = resolution
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    x_grid_m, _ = np.meshgrid(x, y)
    k = 2.0 * np.pi * ANIM_FREQUENCY_HZ / 340.0
    return spl_to_pressure(sound_pressure_level) * np.cos(k * x_grid_m)


def load_positions_csv(path: Path):
    try:
        raw = np.loadtxt(path, delimiter=",")
    except ValueError:
        raw = np.loadtxt(path, delimiter="\t")
    return raw[:, 0].astype(np.float64), raw[:, 1:].astype(np.float64), path.name


def clip_time(times_s, x_mm, t_max):
    mask = times_s <= (t_max + 1e-15)
    if np.count_nonzero(mask) < 2:
        raise RuntimeError(f"Need >=2 samples with t <= {t_max}")
    return times_s[mask], x_mm[mask]


def resample_columns(times_src, values_src, times_dst):
    out = np.empty((times_dst.size, values_src.shape[1]), dtype=np.float64)
    for col in range(values_src.shape[1]):
        out[:, col] = np.interp(times_dst, times_src, values_src[:, col])
    return out


def build_fem_early(fem_csv: Path, t_pinn, x_pinn_mm):
    """COMSOL initial positions with the early-time PINN displacement history."""
    _, x_fem, fem_name = load_positions_csv(fem_csv)
    if x_fem.shape[1] != x_pinn_mm.shape[1]:
        raise RuntimeError(
            f"Particle count mismatch: FEM {x_fem.shape[1]} vs PINN {x_pinn_mm.shape[1]}"
        )
    x_fem0 = x_fem[0, np.argsort(x_fem[0])]
    x_fem_early = x_fem0[None, :] + (x_pinn_mm - x_pinn_mm[0:1, :])
    return t_pinn.copy(), x_fem_early, fem_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fem-csv", type=Path, default=DEFAULT_FEM_CSV)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "comsol_vs_pinn_animation.gif",
    )
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX_S)
    parser.add_argument("--video-seconds", type=float, default=DEFAULT_VIDEO_SECONDS)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--viz-n", type=int, default=20)
    args = parser.parse_args()

    pinn_csv = args.data_dir.resolve() / "pinn_early_1ms.csv"
    fem_csv = args.fem_csv.resolve()
    if not pinn_csv.is_file():
        raise FileNotFoundError(pinn_csv)
    if not fem_csv.is_file():
        raise FileNotFoundError(fem_csv)

    out_path = args.out.resolve()
    t_pinn, x_pinn_mm, pinn_name = load_positions_csv(pinn_csv)
    t_pinn, x_pinn_mm = clip_time(t_pinn, x_pinn_mm, args.t_max)
    x_pinn_mm = x_pinn_mm[:, np.argsort(x_pinn_mm[0])]
    t_fem, x_fem_mm, fem_name = build_fem_early(fem_csv, t_pinn, x_pinn_mm)

    t0 = float(t_pinn[0])
    t1 = min(float(t_pinn[-1]), float(args.t_max))
    save_fps = max(1, int(args.fps))
    n_frames = max(2, int(round(float(args.video_seconds) * save_fps)))
    times_anim = np.linspace(t0, t1, n_frames, dtype=np.float64)
    x_pinn_anim = resample_columns(t_pinn, x_pinn_mm, times_anim)
    x_fem_anim = resample_columns(t_fem, x_fem_mm, times_anim)

    domain_x_mm, domain_y_mm = 34.0, 34.0
    x_grid = np.linspace(2.0, 32.0, 200)
    display_gain = 8.0

    n = x_pinn_anim.shape[1]
    viz_n = min(max(2, int(args.viz_n)), n)
    viz_idx = np.linspace(0, n - 1, viz_n, dtype=np.int64)
    dx = np.linspace(2.5, domain_x_mm - 2.5, viz_n)
    dy = np.linspace(domain_y_mm - 0.5, 0.5, viz_n)
    p_x0 = x_pinn_anim[0, viz_idx]
    f_x0 = x_fem_anim[0, viz_idx]
    c0 = concentration(x_pinn_anim[0], x_grid)

    def disp_x(cur, base, x0):
        return base + (cur - x0) * display_gain

    sample = np.concatenate(
        [
            disp_x(x_pinn_anim[i, viz_idx], dx, p_x0)
            for i in (0, n_frames // 2, n_frames - 1)
        ]
        + [
            disp_x(x_fem_anim[i, viz_idx], dx, f_x0)
            for i in (0, n_frames // 2, n_frames - 1)
        ]
    )
    x_lo, x_hi = float(np.min(sample) - 1.0), float(np.max(sample) + 1.0)

    rel_samples = [
        relative_change(x_pinn_anim[i], c0, x_grid, CURVE_SMOOTH_SIGMA)
        for i in np.linspace(0, n_frames - 1, num=min(40, n_frames), dtype=int)
    ]
    rel = np.concatenate(rel_samples)
    lo, hi = float(np.percentile(rel, 2)), float(np.percentile(rel, 98))
    pad = 0.12 * max(hi - lo, 0.5)
    conc_ylim = (lo - pad, hi + pad)

    sound = cosine_sound_field()
    amp = max(float(np.max(np.abs(sound))), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    (ax11, ax13), (ax21, ax23) = axes

    def setup_row(ax_a, ax_c, title_prefix):
        img = ax_a.imshow(
            sound,
            extent=[0.0, domain_x_mm, 0.0, domain_y_mm],
            origin="lower",
            cmap="RdBu_r",
            interpolation="bilinear",
            aspect="auto",
            alpha=0.35,
            vmin=-amp,
            vmax=amp,
            zorder=0,
            clip_on=False,
        )
        ax_a.set_xlim(x_lo, x_hi)
        ax_a.set_ylim(0.0, domain_y_mm)
        ax_a.set_xlabel("Display X (mm)")
        ax_a.set_ylabel("Display Y (mm)")
        ax_a.grid(True, alpha=0.25)
        ax_a.set_title(f"{title_prefix} particle motion (10 kHz)")
        (line,) = ax_c.plot(
            x_grid, np.zeros_like(x_grid), "r-", lw=2.2, label="Relative change %"
        )
        ax_c.set_xlim(2.0, 32.0)
        ax_c.set_ylim(*conc_ylim)
        ax_c.axvline(8.5, color="0.55", ls="--", lw=1.0, alpha=0.85)
        ax_c.axvline(25.5, color="0.55", ls="--", lw=1.0, alpha=0.85)
        ax_c.set_xlabel("x (mm)")
        ax_c.set_ylabel("Relative Change (%)")
        ax_c.set_title(f"{title_prefix} concentration (10 kHz)")
        ax_c.grid(True, alpha=0.3)
        ax_c.legend(loc="upper right", fontsize=12)
        return img, line

    p_img, p_line = setup_row(ax11, ax13, "FF-PINN")
    f_img, f_line = setup_row(ax21, ax23, "FEM (COMSOL)")

    for ax in (ax11, ax21):
        ax.scatter(
            dx, dy, s=110, c="#1a1a1a", alpha=0.7, edgecolors="white", lw=0.6,
            label="Initial", zorder=1, clip_on=False,
        )
    p_sc = ax11.scatter(
        dx, dy, s=130, c="#00ff66", alpha=0.95, edgecolors="#003300", lw=0.8,
        label="Current", zorder=2, clip_on=False,
    )
    f_sc = ax21.scatter(
        dx, dy, s=130, c="#00ff66", alpha=0.95, edgecolors="#003300", lw=0.8,
        label="Current", zorder=2, clip_on=False,
    )
    ax11.legend(loc="upper right", fontsize=12)
    ax21.legend(loc="upper right", fontsize=12)
    plt.tight_layout()

    def init():
        p_sc.set_offsets(np.column_stack((dx, dy)))
        f_sc.set_offsets(np.column_stack((dx, dy)))
        p_line.set_data(x_grid, np.zeros_like(x_grid))
        f_line.set_data(x_grid, np.zeros_like(x_grid))
        return p_img, p_sc, p_line, f_img, f_sc, f_line

    def update(i):
        t = float(times_anim[i])
        phase = np.cos(2.0 * np.pi * ANIM_FREQUENCY_HZ * t)
        p_img.set_data(sound * phase)
        f_img.set_data(sound * phase)
        px, fx = x_pinn_anim[i], x_fem_anim[i]
        p_sc.set_offsets(np.column_stack((disp_x(px[viz_idx], dx, p_x0), dy)))
        f_sc.set_offsets(np.column_stack((disp_x(fx[viz_idx], dx, f_x0), dy)))
        rel = relative_change(px, c0, x_grid, CURVE_SMOOTH_SIGMA)
        p_line.set_data(x_grid, rel)
        f_line.set_data(x_grid, rel)
        ax11.set_title(f"FF-PINN particle motion (10 kHz)  t = {t * 1e6:.0f} us")
        ax21.set_title(f"FEM (COMSOL) particle motion (10 kHz)  t = {t * 1e6:.0f} us")
        return p_img, p_sc, p_line, f_img, f_sc, f_line

    print(f"PINN: {pinn_name} | FEM: {fem_name}")
    print(f"Physics: {t0 * 1e6:.0f} .. {t1 * 1e6:.0f} us @ {ANIM_FREQUENCY_HZ:.0f} Hz")
    print(f"Playback: {n_frames} frames / {save_fps} FPS (~{n_frames / save_fps:.1f} s)")

    ani = FuncAnimation(
        fig, update, frames=n_frames, init_func=init, interval=1, blit=False, repeat=False
    )
    ani.save(out_path, writer=manimation.PillowWriter(fps=save_fps), dpi=int(args.dpi))
    plt.close(fig)
    print(f"Saved {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
