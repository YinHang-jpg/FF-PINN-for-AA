"""Compose manuscript Fig. 13: top (a)|(b), bottom (c); tight layout, large fonts.

Panel (a): mean orthokinetic kernel vs time for 8/10/12/16 kHz from
    ``panel_a_curves.npz`` (built by ``plot_A.py`` from kernel NPZ archives).
Panel (b): Mednikov-style spatial kernel modulation (cached JSON, else analytic).
Panel (c): diameter-pair heatmaps at 8 kHz and 16 kHz from
    ``panel_c_heatmaps.npz``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "results"))
from _figures import FIGURES  # noqa: E402

HEATMAPS = OUT / "panel_c_heatmaps.npz"
CURVES = OUT / "panel_a_curves.npz"
MEDNIKOV_JSON = OUT / "kernel_mednikov_polydisperse.json"

COLORS = {
    8: ("#1f77b4", "-"),
    10: ("#2ca02c", "--"),
    12: ("#ff7f0e", "-."),
    16: ("#9467bd", "-"),
}
C0, POS_FREQ, DOMAIN = 340.0, 10000.0, 0.034


def _ensure_curves() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not CURVES.is_file():
        import runpy

        runpy.run_path(str(OUT / "plot_A.py"), run_name="__main__")
    d = np.load(CURVES, allow_pickle=True)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for fk in (8, 10, 12, 16):
        out[fk] = (np.asarray(d[f"t_{fk}"], float), np.asarray(d[f"k_{fk}"], float))
    return out


def _load_heatmaps() -> tuple[np.ndarray, np.ndarray]:
    if not HEATMAPS.is_file():
        raise FileNotFoundError(
            f"Missing {HEATMAPS.name}. Required for Fig. 13(c) heatmaps."
        )
    z = np.load(HEATMAPS)
    return np.asarray(z["K_8"], float), np.asarray(z["K_16"], float)


def _draw_heatmap_panel(fig, rect, field: np.ndarray, freq_k: int):
    x0, y0, w, h = rect
    ax = fig.add_axes([x0, y0, w, h])
    ax.imshow(
        field,
        origin="upper",
        extent=[1.0, 4.0, 1.0, 4.0],
        aspect="equal",
        cmap="inferno",
        vmin=0.0,
        vmax=50.0,
        interpolation="bilinear",
    )
    ax.set_xlabel(r"$d_1$ ($\mu$m)", fontsize=18)
    ax.set_ylabel(r"$d_2$ ($\mu$m)", fontsize=18)
    ax.tick_params(labelsize=15, width=1.2, length=5)
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(1.0, 4.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.text(
        0.04,
        0.96,
        f"{freq_k} kHz",
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        color="white",
        va="top",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="0.12", edgecolor="none", alpha=0.55),
    )
    return ax


def _draw_a(ax, curves):
    for fk in (8, 10, 12, 16):
        t, k = curves[fk]
        c, ls = COLORS[fk]
        ax.plot(t, k, color=c, ls=ls, lw=2.8, label=f"{fk} kHz")
    ax.set_xlabel(r"Time ($\mu$s) — one period", fontsize=18)
    ax.set_ylabel(r"Mean kernel [$10^{-12}\,\mathrm{m}^3/\mathrm{s}$]", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 31)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(True, alpha=0.3, ls="--", lw=0.7)
    ax.legend(loc="lower right", fontsize=13, frameon=True, fancybox=False, edgecolor="0.55")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_b(ax):
    cache = MEDNIKOV_JSON
    if cache.is_file():
        import json

        data = json.loads(cache.read_text(encoding="utf-8"))
        x_mm = np.asarray(data["position_x_mm"], dtype=float)
        K = np.asarray(data["K_1_3um_vs_x_m3s"], dtype=float)
        Kn = K / max(float(np.max(K)), 1e-30)
    else:
        xs = np.linspace(0, DOMAIN, 400)
        kk = 2 * np.pi * POS_FREQ / C0
        x_mm, Kn = xs * 1e3, np.abs(np.cos(kk * xs))
    lam = (C0 / POS_FREQ) * 1e3
    ax.plot(x_mm, Kn, color="#1f77b4", lw=3.0, label=r"$K_{\mathrm{OI}}(x)/K_{\max}=U_g(x)/U_{g0}$")
    for n in range(8):
        xu = (2 * n + 1) * lam / 4.0
        if 0 <= xu <= DOMAIN * 1e3:
            ax.axvline(xu, color="#2ca02c", ls=":", lw=1.3, zorder=0)
    for n in range(8):
        xa = n * lam / 2.0
        if 0 <= xa <= DOMAIN * 1e3:
            ax.axvline(xa, color="#d62728", ls="--", lw=1.1, alpha=0.75, zorder=0)
    ax.set_xlabel(r"Position $x$ (mm)", fontsize=18)
    ax.set_ylabel(r"Normalized kernel $K/K_{\max}$", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.set_xlim(0, DOMAIN * 1e3)
    ax.set_ylim(0, 1.12)
    ax.grid(True, alpha=0.3, ls="--", lw=0.7)
    ax.legend(loc="upper right", fontsize=12, frameon=True, fancybox=False, edgecolor="0.55")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_tag(ax, letter: str):
    ax.text(
        0.02,
        0.97,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9),
    )


def main():
    curves = _ensure_curves()
    f8, f16 = _load_heatmaps()

    fig = plt.figure(figsize=(12.0, 9.2), facecolor="white")
    ax_a = fig.add_axes([0.07, 0.62, 0.42, 0.34])
    ax_b = fig.add_axes([0.55, 0.62, 0.41, 0.34])
    _draw_a(ax_a, curves)
    _draw_b(ax_b)
    _panel_tag(ax_a, "a")
    _panel_tag(ax_b, "b")

    _draw_heatmap_panel(fig, [0.08, 0.07, 0.34, 0.40], f8, 8)
    _draw_heatmap_panel(fig, [0.50, 0.07, 0.34, 0.40], f16, 16)

    fig.add_artist(
        plt.Line2D(
            [0.04, 0.96],
            [0.525, 0.525],
            transform=fig.transFigure,
            color="0.55",
            lw=1.15,
            solid_capstyle="round",
            clip_on=False,
            zorder=0,
        )
    )
    fig.text(0.02, 0.48, "(c)", fontsize=18, fontweight="bold", va="bottom", ha="left")

    cax = fig.add_axes([0.88, 0.10, 0.02, 0.34])
    sm = ScalarMappable(norm=Normalize(0.0, 50.0), cmap="inferno")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$K$ [$10^{-12}\,\mathrm{m}^3/\mathrm{s}$]", fontsize=16)
    cb.ax.tick_params(labelsize=14)
    cb.set_ticks([0, 10, 20, 30, 40, 50])

    png = FIGURES / "fig13_kernel_abc.png"
    fig.savefig(png, dpi=400, facecolor="white")
    plt.close(fig)
    print("WROTE", png)
    for fk in (8, 10, 12, 16):
        t, k = curves[fk]
        print(f"  {fk}k: max={k.max():.2f} at t={t[np.argmax(k)]:.1f}")


if __name__ == "__main__":
    main()
