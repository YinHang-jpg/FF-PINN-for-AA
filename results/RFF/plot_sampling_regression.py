"""
R3Q5 figure: two panels (MSE | training time).

Smooth monotone parametric curves only (no interpolation through points).
MSE:  a + b N^{-c}   (a,b,c > 0) — monotone decreasing
Time:  a + b N        (b > 0)     — monotone increasing
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

OUT = Path(__file__).resolve().parent
if str(OUT.parents[1] / "results") not in sys.path:
    sys.path.insert(0, str(OUT.parents[1] / "results"))
from _figures import FIGURES  # noqa: E402
JSON_PATH = OUT / "sampling_points_sensitivity.json"

CASE_STYLE = {
    4: {"color": "tab:blue", "ls": "-"},
    10: {"color": "tab:orange", "ls": "--"},
    16: {"color": "tab:purple", "ls": "-."},
    18: {"color": "tab:red", "ls": ":"},
    20: {"color": "tab:green", "ls": "-"},
}


def _mse_model(n, a, b, c):
    return a + b * np.power(n, -c)


def _fit_mse_powerlaw(x, y):
    """Fit y = a + b N^{-c} (a,b,c > 0) by grid search on c + WLS on (a,b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y = np.clip(y, 1e-30, None)
    w = 1.0 / y
    best = None
    for c in np.linspace(0.6, 4.5, 80):
        z = np.power(x, -c)
        A = np.column_stack([np.ones_like(z), z])
        Aw, yw = A * w[:, None], y * w
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if a < 0:
            a = max(float(y.min()) * 0.05, 1e-12)
            b = float(np.sum(w * z * (y - a)) / np.sum(w * z * z))
        if b <= 0:
            continue
        pred = _mse_model(x, a, b, c)
        if np.any(pred <= 0):
            continue
        err = float(np.sum((np.log(pred) - np.log(y)) ** 2))
        if best is None or err < best[0]:
            best = (err, a, b, float(c))
    if best is None:
        slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
        return 1e-12, float(np.exp(intercept)), float(max(-slope, 0.8))
    return best[1], best[2], best[3]


def regress_mse(x, y, *, n_dense=400):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b, c = _fit_mse_powerlaw(x, y)
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), n_dense)
    yd = _mse_model(xd, a, b, c)
    return xd, yd


def regress_time(x, y, *, n_dense=400):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(max(coef[1], 0.0))
    xd = np.logspace(np.log10(x.min()), np.log10(x.max()), n_dense)
    yd = np.maximum(a + b * xd, 0.0)
    return xd, yd


def series_by_freq(rows):
    by_f = {}
    for r in rows:
        by_f.setdefault(int(r["freq_khz"]), []).append(r)
    for f in by_f:
        by_f[f] = sorted(by_f[f], key=lambda r: r["N"])
    return dict(sorted(by_f.items()))


def plot_panel(ax, by_f, y_key, ylabel, title, *, regressor, log_y=False):
    for fk, rows in by_f.items():
        style = CASE_STYLE.get(fk, {"color": "0.3", "ls": "-"})
        ns = np.array([r["N"] for r in rows], dtype=float)
        ys = np.array([r[y_key] for r in rows], dtype=float)
        xd, yd = regressor(ns, ys)
        ax.plot(xd, yd, color=style["color"], ls=style["ls"], lw=2.2, label=f"{fk} kHz")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of training samples N")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(20000, color="0.55", ls=":", lw=1.1, zorder=1)
    ax.legend(fontsize=8.5, framealpha=0.92)


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    by_f = series_by_freq(data["results"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))
    plot_panel(
        ax1,
        by_f,
        "mse_norm",
        r"MSE / $|F|_{\mathrm{peak}}^{2}$",
        "(a)",
        regressor=regress_mse,
        log_y=True,
    )
    plot_panel(
        ax2,
        by_f,
        "train_s",
        "Training time (s)",
        "(b)",
        regressor=regress_time,
    )
    fig.tight_layout()
    png = FIGURES / "fig03_sampling_points_sensitivity.png"
    pdf = FIGURES / "fig03_sampling_points_sensitivity.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("WROTE", png.name)


if __name__ == "__main__":
    main()
