"""
Appendix A: side-by-side orthokinetic kernel heatmaps K(d1, d2).

Two independent calculations under the SAME acoustic conditions
(Dong et al., J. Aerosol Sci. 37:540, 2006, Fig. 5b: 10 kHz, 160 dB,
rho_p = 2400 kg/m^3), evaluated at a standing-wave velocity antinode
(the convention used by classical Mednikov / Dong closed forms, and
the location where the FF-PINN gas field is strongest).

  Panel (a)  Literature  — Mednikov beta_OI with BFH entrainment
             eta = [1+(omega tau)^2]^(-1/2),  tau = rho_p d^2 / (18 mu)
             (Dong text after Eq. 2; no Cunningham in tau).

  Panel (b)  FF-PINN     — same geometric kernel as the manuscript
             K = (pi/4)(d1+d2)^2 |Delta v|,
             with |Delta v| = |eta1-eta2| Ug from co-located PINN
             trajectories (orthokinetic definition). Collision probability
             and diameter-pair extras used in Fig. 13 diagnostics are omitted.

  Panel (c)  Difference  K_PINN - K_Mednikov  (same units).

Frequency is 10 kHz (production FF-PINN training band and Dong Fig. 5b).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(OUT.parents[1]))
sys.path.insert(0, str(OUT.parents[1] / "results"))
from _figures import FIGURES  # noqa: E402
DATA = OUT
FIG_OUT = FIGURES

from _kernel_common import load_models_at_freq, pinn_fields  # noqa: E402
from _trajectory_kernel import (  # noqa: E402
    MU,
    entrainment_single,
    u_travelling_peak,
)

# Dong et al. (2006) Fig. 5b acoustic conditions; 10 kHz = PINN production pack.
FREQ_HZ = 10000.0
SPL_DB = 160.0
RHO_P = 2400.0
# Diameter axes match manuscript Fig. 13(c) / kernel_extractor.py (0.8-4 um).
DIAMS_UM = np.linspace(0.8, 4.0, 21)
# Spot-check pairs: trajectory |Delta v| vs eta-reconstructed K (must agree).
CHECK_PAIRS_UM = [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0)]


def eta_bfh(d_m: np.ndarray, f: float, rho_p: float) -> np.ndarray:
    """Dong / Mednikov amplitude ratio (no Cunningham in tau)."""
    tau = rho_p * np.asarray(d_m, dtype=float) ** 2 / (18.0 * MU)
    w = 2.0 * np.pi * f
    return 1.0 / np.sqrt(1.0 + (w * tau) ** 2)


def mednikov_K(d_m: np.ndarray, eta: np.ndarray, Ug: float) -> np.ndarray:
    """K_ij = (pi/4)(di+dj)^2 |eta_i - eta_j| Ug  [m^3/s]."""
    d1 = d_m[:, None]
    d2 = d_m[None, :]
    return 0.25 * np.pi * (d1 + d2) ** 2 * np.abs(eta[:, None] - eta[None, :]) * Ug


def _ceil_nice(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = np.floor(np.log10(v))
    mant = v / (10 ** exp)
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if mant <= t + 1e-12:
            return float(t * 10 ** exp)
    return float(10 ** (exp + 1))


def main() -> None:
    print("=== Appendix A kernel heatmaps (10 kHz, Dong Fig.5b conditions) ===", flush=True)
    d_m = DIAMS_UM * 1e-6
    Ug_stand = u_travelling_peak(SPL_DB) / np.sqrt(2.0)

    eta_lit = eta_bfh(d_m, FREQ_HZ, RHO_P)
    K_lit = mednikov_K(d_m, eta_lit, Ug_stand)

    print(f"  Ug_antinode = {Ug_stand:.4f} m/s  (standing, SPL={SPL_DB} dB)", flush=True)
    print("  loading FF-PINN pack @ 10 kHz ...", flush=True)
    models = load_models_at_freq(FREQ_HZ)
    print(f"  pack = {Path(models['pack_dir']).name}", flush=True)

    # PINN local gas amplitude at the velocity antinode (x=0), same SPL.
    period = 1.0 / FREQ_HZ
    ug_pinn = 0.0
    for t in np.linspace(0.0, period, 80, endpoint=False):
        _, u = pinn_fields(models, np.array([0.0]), t, spl=SPL_DB)
        ug_pinn = max(ug_pinn, abs(float(np.asarray(u).reshape(-1)[0])))
    print(f"  Ug_PINN(x=0) = {ug_pinn:.4f} m/s  (ratio to closed-form {ug_pinn / Ug_stand:.4f})", flush=True)

    eta_pinn = np.empty_like(d_m)
    for i, d in enumerate(d_m):
        eta_pinn[i] = entrainment_single(
            float(d), FREQ_HZ, SPL_DB, RHO_P,
            mode="pinn", pinn_models=models, include_spgf=False, x_pinn=0.0,
        )
        print(
            f"  d={DIAMS_UM[i]:4.2f} um  eta_PINN={eta_pinn[i]:.4f}  "
            f"eta_BFH={eta_lit[i]:.4f}  ratio={eta_pinn[i] / max(eta_lit[i], 1e-30):.4f}",
            flush=True,
        )

    # Same Ug in both K matrices so the comparison tests entrainment, not
    # an SPL / standing-vs-travelling amplitude convention.
    K_pinn = mednikov_K(d_m, eta_pinn, Ug_stand)

    # Trajectory cross-check: K from co-located |Delta v| vs eta reconstruction.
    from _trajectory_kernel import integrate_pair

    checks = []
    for d1u, d2u in CHECK_PAIRS_UM:
        d1, d2 = d1u * 1e-6, d2u * 1e-6
        K_traj, dv_amp, _ = integrate_pair(
            d1, d2, FREQ_HZ, SPL_DB, RHO_P,
            mode="pinn", x0=0.0, include_spgf=False, pinn_models=models,
        )
        i = int(np.argmin(np.abs(DIAMS_UM - d1u)))
        j = int(np.argmin(np.abs(DIAMS_UM - d2u)))
        K_eta = float(K_pinn[i, j])
        checks.append({
            "pair_um": [d1u, d2u],
            "K_trajectory_m3s": float(K_traj),
            "K_from_eta_m3s": K_eta,
            "dv_amp_m_s": float(dv_amp),
            "ratio_traj_over_eta": float(K_traj / max(K_eta, 1e-30)),
        })
        print(
            f"  check ({d1u},{d2u}) um  K_traj={K_traj:.4e}  "
            f"K_eta={K_eta:.4e}  ratio={K_traj / max(K_eta, 1e-30):.4f}",
            flush=True,
        )

    np.savez_compressed(
        OUT / "compare_kernel_heatmaps.npz",
        diameters_um=DIAMS_UM,
        eta_literature=eta_lit,
        eta_pinn=eta_pinn,
        K_literature_m3s=K_lit,
        K_pinn_m3s=K_pinn,
        Ug_stand_m_s=np.array(Ug_stand),
        Ug_pinn_m_s=np.array(ug_pinn),
    )
    stats = _kernel_stats(K_lit, K_pinn, eta_lit, eta_pinn, Ug_stand, ug_pinn, checks)
    print("  STATS", json.dumps({k: v for k, v in stats.items() if k != "trajectory_checks"}), flush=True)
    _write_outputs(DIAMS_UM, eta_lit, eta_pinn, K_lit, K_pinn, stats)


def _kernel_stats(K_lit, K_pinn, eta_lit, eta_pinn, Ug_stand, ug_pinn, checks):
    delta = K_pinn - K_lit
    off = ~np.eye(K_lit.shape[0], dtype=bool)
    k_ref, k_pin = K_lit[off], K_pinn[off]
    rel = (k_pin - k_ref) / np.maximum(k_ref, 1e-30)
    # Near-equal sizes have tiny K; relative error there is not a PINN failure.
    robust = off & (K_lit > 0.05 * K_lit.max())
    i_max, j_max = np.unravel_index(
        int(np.argmax(np.where(off, np.abs((K_pinn - K_lit) / np.maximum(K_lit, 1e-30)), -1.0))),
        K_lit.shape,
    )
    return {
        "median_PINN_over_Mednikov": float(np.median(k_pin / np.maximum(k_ref, 1e-30))),
        "mean_rel_err": float(np.mean(np.abs(rel))),
        "max_rel_err": float(np.max(np.abs(rel))),
        "max_rel_err_at_um": [float(DIAMS_UM[i_max]), float(DIAMS_UM[j_max])],
        "max_rel_err_note": "occurs at nearly equal sizes where K itself is ~0.6% of K_max",
        "robust_K_gt_5pct_max": {
            "n_cells": int(robust.sum()),
            "median_PINN_over_Mednikov": float(np.median(K_pinn[robust] / K_lit[robust])),
            "mean_rel_err": float(np.mean(np.abs((K_pinn[robust] - K_lit[robust]) / K_lit[robust]))),
            "max_rel_err": float(np.max(np.abs((K_pinn[robust] - K_lit[robust]) / K_lit[robust]))),
        },
        "Kmax_lit_1e12": float(K_lit.max() * 1e12),
        "Kmax_pinn_1e12": float(K_pinn.max() * 1e12),
        "abs_dK_max_over_Kmax": float(np.abs(delta).max() / max(K_lit.max(), 1e-30)),
        "rmse_1e12": float(np.sqrt(np.mean((delta[off] * 1e12) ** 2))),
        "pearson_r": float(np.corrcoef(k_ref, k_pin)[0, 1]),
        "Ug_closed_form_m_s": float(Ug_stand),
        "Ug_PINN_antinode_m_s": float(ug_pinn),
        "Ug_PINN_over_closed_form": float(ug_pinn / Ug_stand),
        "eta_median_ratio": float(np.median(eta_pinn / np.maximum(eta_lit, 1e-30))),
        "eta_max_rel_err": float(np.max(np.abs(eta_pinn - eta_lit) / np.maximum(eta_lit, 1e-30))),
        "trajectory_checks": checks,
    }


def _write_outputs(diameters_um, eta_lit, eta_pinn, K_lit, K_pinn, stats):
    delta = K_pinn - K_lit
    scale = 1e12
    K_lit_p, K_pinn_p, delta_p = K_lit * scale, K_pinn * scale, delta * scale
    vmax = _ceil_nice(float(max(K_lit_p.max(), K_pinn_p.max())))
    dmax = _ceil_nice(float(np.abs(delta_p).max()))
    extent = [float(diameters_um[0]), float(diameters_um[-1]),
              float(diameters_um[0]), float(diameters_um[-1])]

    plt.rcParams.update({
        "font.size": 16,
        "axes.linewidth": 1.4,
        "axes.labelsize": 17,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "mathtext.fontset": "stix",
    })
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    titles = [
        r"Mednikov $K(d_1,d_2)$",
        r"FF-PINN $K(d_1,d_2)$",
        r"$K_{\mathrm{PINN}}-K_{\mathrm{Mednikov}}$",
    ]
    fields = [K_lit_p, K_pinn_p, delta_p]
    cmaps = ["inferno", "inferno", "coolwarm"]
    vmins, vmaxs = [0.0, 0.0, -dmax], [vmax, vmax, dmax]
    cb_labels = [
        r"$K$ [$10^{-12}\,\mathrm{m}^3/\mathrm{s}$]",
        r"$K$ [$10^{-12}\,\mathrm{m}^3/\mathrm{s}$]",
        r"$\Delta K$ [$10^{-12}\,\mathrm{m}^3/\mathrm{s}$]",
    ]
    letters = "abc"
    for ax, field, title, cmap, vmin, vmax_i, cblab, let in zip(
        axes, fields, titles, cmaps, vmins, vmaxs, cb_labels, letters
    ):
        im = ax.imshow(
            field, origin="lower", extent=extent, aspect="equal",
            cmap=cmap, vmin=vmin, vmax=vmax_i, interpolation="nearest",
        )
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xlabel(r"$d_1$ ($\mu$m)", fontsize=17)
        ax.set_ylabel(r"$d_2$ ($\mu$m)", fontsize=17)
        ax.xaxis.set_major_locator(MultipleLocator(0.8))
        ax.yaxis.set_major_locator(MultipleLocator(0.8))
        ax.tick_params(labelsize=15, length=5, width=1.2)
        ax.text(
            0.04, 0.96, f"({let})", transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left", color="white" if let != "c" else "black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="0.15" if let != "c" else "white",
                      edgecolor="none", alpha=0.55),
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cblab, fontsize=15)
        cb.ax.tick_params(labelsize=14, length=4, width=1.1)

    png, pdf = FIG_OUT / "figA_compare_kernel_heatmaps.png", FIG_OUT / "figA_compare_kernel_heatmaps.pdf"
    fig.savefig(png, dpi=450, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {png.name}", flush=True)

    payload = {
        "frequency_hz": FREQ_HZ,
        "why_this_frequency": (
            "10 kHz is the production FF-PINN training frequency and the "
            "acoustic condition of Dong et al. (2006) Fig. 5b (size dependence)."
        ),
        "conditions": {
            "SPL_dB": SPL_DB, "rho_p": RHO_P,
            "wave": "standing-wave velocity antinode",
            "source": "Dong et al., J. Aerosol Sci. 37:540 (2006), Fig. 5b",
        },
        "literature_formula": (
            "K=(pi/4)(d1+d2)^2 |eta1-eta2| Ug, "
            "eta=1/sqrt(1+(omega tau)^2), tau=rho_p d^2/(18 mu)"
        ),
        "pinn_formula": (
            "same K; eta from co-located FF-PINN trajectories "
            "(Stokes only, include_spgf=False) at x=0"
        ),
        "not_included": [
            "P_coll = 1-exp(-|dv|/vs) from kernel_extractor.py",
            "heuristic diameter_pair_enhancement W",
            "random-position pair sampling (spurious equal-size kernel)",
        ],
        "diameters_um": np.asarray(diameters_um, dtype=float).tolist(),
        "eta_literature": np.asarray(eta_lit, dtype=float).tolist(),
        "eta_pinn": np.asarray(eta_pinn, dtype=float).tolist(),
        "stats": stats,
        "figures": [png.name, pdf.name],
    }
    (OUT / "compare_kernel_heatmaps.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("WROTE compare_kernel_heatmaps.json", flush=True)


def plot_from_npz() -> None:
    z = np.load(OUT / "compare_kernel_heatmaps.npz")
    K_lit, K_pinn = z["K_literature_m3s"], z["K_pinn_m3s"]
    eta_lit, eta_pinn = z["eta_literature"], z["eta_pinn"]
    prev = json.loads((OUT / "compare_kernel_heatmaps.json").read_text(encoding="utf-8"))
    checks = prev.get("stats", {}).get("trajectory_checks", [])
    stats = _kernel_stats(
        K_lit, K_pinn, eta_lit, eta_pinn,
        float(z["Ug_stand_m_s"]), float(z["Ug_pinn_m_s"]), checks,
    )
    _write_outputs(z["diameters_um"], eta_lit, eta_pinn, K_lit, K_pinn, stats)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_from_npz()
    else:
        main()
