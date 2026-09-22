"""Combined 2x2 figure for manuscript Fig. 2.

(a) training loss vs epoch for M = 8/16/32/48/64 (target MSE = 1e-8)
(b) epochs to reach the target vs M (M=0..64 step 8); M=0 shown as cap bar
(c) temporal factor without Fourier features
(d) temporal factor with Fourier features (M = 32)

Reads:  fourier_M_panelA_target1e-8.json, fourier_M_sweep0_64_step8.json,
        fourier_M_panel_cd.npz
Writes: figures/fig02_fourier_M_composite.png / .pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
sys_path_root = OUT_DIR.parents[1]
import sys

if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root / "results"))
from _figures import FIGURES  # noqa: E402

PANEL_A_JSON = OUT_DIR / "fourier_M_panelA_target1e-8.json"
SWEEP_JSON = OUT_DIR / "fourier_M_sweep0_64_step8.json"
CD_NPZ = OUT_DIR / "fourier_M_panel_cd.npz"
PNG_PATH = FIGURES / "fig02_fourier_M_composite.png"
PDF_PATH = FIGURES / "fig02_fourier_M_composite.pdf"

TARGET_LOSS = 1e-8
RECOMMENDED_M = 32
M_ORDER = [8, 16, 32, 48, 64]
M_COLORS = {
    0:  "0.55",
    8:  "tab:olive",
    16: "tab:green",
    24: "tab:cyan",
    32: "tab:blue",
    40: "tab:orange",
    48: "tab:red",
    56: "tab:pink",
    64: "tab:purple",
}
THEORY_C = "#1f4e79"
PINN_C = "#c0392b"


def _sqrt_fwd(v):
    return np.sqrt(np.clip(np.asarray(v, dtype=float), 0.0, None))


def _sqrt_inv(v):
    return np.square(np.clip(np.asarray(v, dtype=float), 0.0, None))


def load_loss_histories() -> dict[int, np.ndarray]:
    data = json.loads(PANEL_A_JSON.read_text(encoding="utf-8"))
    return {int(m["M"]): np.asarray(m["loss_hist"], dtype=float) for m in data["models"]}


def load_cd_curves() -> dict[str, np.ndarray]:
    """Load panel (c)/(d) curves from ``fourier_M_panel_cd.npz``."""
    if not CD_NPZ.is_file():
        raise FileNotFoundError(CD_NPZ)
    z = np.load(CD_NPZ, allow_pickle=True)
    required = ("t_c", "y_c", "th_c", "t_d", "y_d", "th_d")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{CD_NPZ.name} missing keys: {missing}")
    return {k: np.asarray(z[k], dtype=float) for k in required}


def panel_a(ax, histories: dict[int, np.ndarray]) -> None:
    for M in M_ORDER:
        h = histories[M]
        lw = 2.6 if M == RECOMMENDED_M else 1.8
        ax.plot(h[:, 0] / 1e3, h[:, 1], color=M_COLORS[M], lw=lw, label=rf"$M$ = {M}")
    ax.axhline(TARGET_LOSS, color="0.40", ls="--", lw=1.4, label=r"Target $10^{-8}$")
    ax.set_xscale("function", functions=(_sqrt_fwd, _sqrt_inv))
    ax.set_yscale("log")
    ax.set_xlabel(r"Training epoch ($\times 10^{3}$)", fontweight="bold")
    ax.set_ylabel("MSE loss", fontweight="bold")
    ax.set_ylim(TARGET_LOSS / 3.0, 2.0)
    ax.set_xlim(0, 400)
    ax.set_xticks([0, 5, 20, 50, 100, 200, 300, 400])
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2)


def _load_sweep_points() -> list[dict]:
    data = json.loads(SWEEP_JSON.read_text(encoding="utf-8"))
    return sorted(data["models"], key=lambda m: m["M"])


def panel_b(ax) -> None:
    models = _load_sweep_points()
    M_ORDER_B = [0, 8, 16, 24, 32, 40, 48, 56, 64]

    by_M = {m["M"]: m for m in models}
    reached_eps = [by_M[M]["epochs"] for M in M_ORDER_B
                   if M in by_M and by_M[M]["reached_target"]]
    min_ep = min(reached_eps) if reached_eps else 1e4
    max_ep = max(reached_eps) if reached_eps else 5e5

    y_base = min_ep * 0.15
    cap_height = max_ep * 2.0
    bar_width = 6.0

    for M in M_ORDER_B:
        if M not in by_M:
            continue
        rec = by_M[M]
        is_rec = (M == RECOMMENDED_M)
        if not rec["reached_target"]:
            ax.bar(
                M, cap_height, bottom=y_base,
                width=bar_width,
                color="0.88", edgecolor="black",
                hatch="//", linewidth=1.0, zorder=3,
                label=r"$M$ = 0 (not converged)",
            )
            ax.text(M, cap_height * 1.08, ">cap", ha="center", va="bottom",
                    fontsize=9.5, color="0.35", fontweight="bold")
        else:
            ep = rec["epochs"]
            ax.bar(
                M, ep - y_base, bottom=y_base,
                width=bar_width,
                color=M_COLORS.get(M, "0.5"),
                edgecolor="black",
                linewidth=1.6 if is_rec else 1.0,
                alpha=1.0 if is_rec else 0.92,
                zorder=3,
                label=rf"$M$ = {M}" + (" (selected)" if is_rec else ""),
            )

    ax.axvline(RECOMMENDED_M, color="tab:blue", ls="--", lw=1.3, alpha=0.55, zorder=1)

    ax.set_yscale("log")
    ax.set_xlabel(r"Fourier feature dimension $M$", fontweight="bold")
    ax.set_ylabel(r"Epochs to reach MSE $=10^{-8}$", fontweight="bold")
    ax.set_xticks(M_ORDER_B)
    ax.set_xlim(-4, 68)
    ax.set_ylim(y_base, cap_height * 8.0)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2, fontsize=10.5)


def panel_cd(ax, t, theory, pinn, pinn_lw: float, dashes: tuple[float, float]) -> None:
    ax.plot(t, theory, color=THEORY_C, lw=2.6, solid_capstyle="round", label="Theory", zorder=2)
    ax.plot(t, pinn, color=PINN_C, lw=pinn_lw, ls="--", dashes=dashes,
            label="PINN prediction", zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.18, 1.18)
    ax.set_xlabel(r"Time ($\mu$s)", fontweight="bold")
    ax.set_ylabel(r"Time factor", fontweight="bold")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", framealpha=0.95)


def main() -> None:
    histories = load_loss_histories()
    cd = load_cd_curves()

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12.5,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "axes.linewidth": 1.1,
            "lines.antialiased": True,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.4))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    panel_a(ax_a, histories)
    panel_b(ax_b)
    panel_cd(ax_c, cd["t_c"], cd["th_c"], cd["y_c"], pinn_lw=1.2, dashes=(2.8, 1.3))
    panel_cd(ax_d, cd["t_d"], cd["th_d"], cd["y_d"], pinn_lw=1.7, dashes=(3.4, 1.7))

    for ax, tag in zip((ax_a, ax_b, ax_c, ax_d), ("(a)", "(b)", "(c)", "(d)")):
        ax.set_title(tag, loc="left", fontweight="bold", pad=8)
        ax.tick_params(width=1.1, length=5.5)

    fig.tight_layout(pad=1.6, h_pad=2.4, w_pad=2.4)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white", dpi=600)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {PNG_PATH.name} / {PDF_PATH.name}")


if __name__ == "__main__":
    main()
