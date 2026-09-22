"""Appendix A literature comparison.

Compare FF-PINN / DEM orthokinetic kernels against closed-form literature
expressions (Mednikov / Dong / Gonzalez).

Each panel separates:
  * Literature curve   — closed-form equations from the cited paper.
  * DEM trajectory     — project mechanisms (Stokes_drag + optional ARF/SPGF).
  * FF-PINN trajectory — unified PINN force (_kernel_common); valid at the
                         trained 10 kHz pack.
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
import numpy as np

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(DATA.parents[1]))
sys.path.insert(0, str(DATA.parents[1] / "results"))
from _figures import FIGURES  # noqa: E402
OUT = FIGURES  # manuscript figures

from _trajectory_kernel import (  # noqa: E402
    MU, RHO0, PREF, integrate_pair, entrainment_single, u_travelling_peak,
    relaxation_time_from_trajectory,
)
from _kernel_common import (  # noqa: E402
    LAMBDA_G, load_models, load_models_at_freq, force_for_diameter, pinn_fields,
    u0_amplitude, DOMAIN, FREQ as FREQ_PINN, RHO_P as RHO_PINN, pack_dir_for_freq,
)
from initialization.sound_source_standing import sound_pressure_level as SPL_PROJECT  # noqa: E402

# ---- literature closed forms (same as before, documented) ----

def Cc(d):
    """Dong Eq.(11) form used only for wake/settling: Cc = 1 + 2.52 λ/d."""
    d = np.asarray(d, dtype=float)
    return 1.0 + 2.52 * LAMBDA_G / d


def tau_p(d, rho_p):
    """
    Dong particle relaxation time (text after Eq.2): τ = ρ_p d² / (18 μ).
    No Cunningham factor here — Dong applies Cc only in settling Eqs.(11)–(12).
    """
    return rho_p * np.asarray(d, dtype=float) ** 2 / (18.0 * MU)


def eta_bfh(d, f, rho_p):
    """Amplitude ratio H = 1/sqrt(1+(w tau)^2) (BFH)."""
    w = 2 * np.pi * f
    return 1.0 / np.sqrt(1.0 + (w * tau_p(d, rho_p)) ** 2)


def eta12_dong(d1, d2, f, rho_p):
    """Dong (2006) Eq.(5): relative entrainment amplitude η12 (≠ |H1-H2|)."""
    w = 2 * np.pi * f
    t1, t2 = float(tau_p(d1, rho_p)), float(tau_p(d2, rho_p))
    return w * abs(t1 - t2) / np.sqrt((1.0 + (w * t1) ** 2) * (1.0 + (w * t2) ** 2))


def mednikov_K(d1, d2, f, spl, rho_p, Ug=None):
    """Classical orthokinetic kernel using |H1-H2| (Zhang/Mednikov style)."""
    if Ug is None:
        Ug = u_travelling_peak(spl)
    return (np.pi / 4.0) * (d1 + d2) ** 2 * np.abs(eta_bfh(d1, f, rho_p) - eta_bfh(d2, f, rho_p)) * Ug


def dong_eps(d1, d2, f, spl, rho_p):
    """Dong Eq.(8)–(9): collision efficiency ε = [St/(St+0.65)]^3.7."""
    Ug = u_travelling_peak(spl)
    n12 = eta12_dong(d1, d2, f, rho_p)
    dL, dS = max(float(d1), float(d2)), min(float(d1), float(d2))
    St = rho_p * n12 * Ug * dS**2 / (18.0 * MU * dL)
    return (max(St, 0.0) / (St + 0.65)) ** 3.7


def dong_Leff_ortho(d1, d2, f, spl, rho_p):
    """Dong Eq.(6)+(10): L_orth^eff = ε * η12 * Ug / ω."""
    w = 2 * np.pi * f
    Ug = u_travelling_peak(spl)
    return eta12_dong(d1, d2, f, rho_p) * Ug / w * dong_eps(d1, d2, f, spl, rho_p)


def dong_Leff(d1, d2, f, spl, rho_p):
    """Alias used by frequency panel."""
    return dong_Leff_ortho(d1, d2, f, spl, rho_p)


def dong_Leff_wake(d1, d2, f, spl, rho_p, g=9.81):
    """
    Dong Eq.(13)–(17): acoustic-wake effective agglomeration length (Dianov).
    hi made dimensionless with ω (paper OCR omits ω; required for consistency).
    Returns NaN when |d1-d2| is tiny (paper: monodisperse undefined with gravity).
    """
    w = 2 * np.pi * f
    Ug = u_travelling_peak(spl)
    d1, d2 = float(d1), float(d2)
    if abs(d1 - d2) < 1e-9:
        return np.nan
    ls = []
    for di in (d1, d2):
        ti = float(tau_p(di, rho_p))
        ni = w * ti / np.sqrt(1.0 + (w * ti) ** 2)  # Eq.(15)
        # Eq.(14): hi = 9 Ug ρg / (π ω di ρp); OCR sometimes drops ω
        hi = 9.0 * Ug * RHO0 / (np.pi * w * di * rho_p)
        li = ni / np.sqrt(1.0 + 2.0 * hi * ni**2 + hi**2 * ni**4)
        ls.append(li)
    l1, l2 = ls
    cc = float(Cc(min(d1, d2)))
    # Eq.(17): L_awe^eff = sqrt[ (3 Ug / 2π) (d1 l1 + d2 l2) * 9μ / (Cc ρp g |d1-d2|) ]
    inside = (
        (3.0 * Ug / (2.0 * np.pi))
        * (d1 * l1 + d2 * l2)
        * (9.0 * MU)
        / (cc * rho_p * g * abs(d1 - d2))
    )
    return float(np.sqrt(max(inside, 0.0)))


def eta12_from_entrainment(e1, e2, f):
    """
    Dong Eq.(5) η12 recovered from measured entrainment factors.
    Inverts η = 1/sqrt(1+(ωτ)^2) → τ, then applies Eq.(5).
    Do NOT use |η1-η2|: at low f that underestimates η12 by ~2–3× (phase).
    """
    w = 2.0 * np.pi * f

    def tau_of(e):
        e = float(np.clip(e, 1e-12, 1.0 - 1e-15))
        return np.sqrt(max(1.0 / e**2 - 1.0, 0.0)) / w

    t1, t2 = tau_of(e1), tau_of(e2)
    return w * abs(t1 - t2) / np.sqrt((1.0 + (w * t1) ** 2) * (1.0 + (w * t2) ** 2))


def dong_Leff_dem(d1, d2, f, spl, rho_p, mode="stokes"):
    """DEM-derived L_orth^eff: η from Stokes trajectory → Dong Eqs.(5)–(10)."""
    w = 2 * np.pi * f
    Ug = u_travelling_peak(spl)
    e1 = entrainment_single(d1, f, spl, rho_p, mode=mode)
    e2 = entrainment_single(d2, f, spl, rho_p, mode=mode)
    n12 = eta12_from_entrainment(e1, e2, f)
    dL, dS = max(float(d1), float(d2)), min(float(d1), float(d2))
    St = rho_p * n12 * Ug * dS**2 / (18.0 * MU * dL)
    eps = (max(St, 0.0) / (St + 0.65)) ** 3.7
    return n12 * Ug / w * eps


def _L_orth_from_eta(dL, d, eta_L, eta_S, f, spl, rho):
    """Dong Eqs.(5)–(10) using entrainment factors (PINN or DEM)."""
    if abs(d - dL) < 1e-12:
        return 0.0
    w = 2 * np.pi * f
    Ug = u_travelling_peak(spl)
    n12 = eta12_from_entrainment(eta_L, eta_S, f)
    dS, dLg = min(d, dL), max(d, dL)
    St = rho * n12 * Ug * dS**2 / (18.0 * MU * dLg)
    eps = (max(St, 0.0) / (St + 0.65)) ** 3.7
    return n12 * Ug / w * eps


def _L_orth_from_tau(d1, d2, tau1, tau2, f, spl, rho):
    """Dong Eqs.(5)–(10) from relaxation times (phase-stable for PINN packs)."""
    if abs(float(d1) - float(d2)) < 1e-12:
        return 0.0
    w = 2 * np.pi * f
    Ug = u_travelling_peak(spl)
    t1, t2 = float(tau1), float(tau2)
    n12 = w * abs(t1 - t2) / np.sqrt((1.0 + (w * t1) ** 2) * (1.0 + (w * t2) ** 2))
    dS, dL = min(float(d1), float(d2)), max(float(d1), float(d2))
    St = rho * n12 * Ug * dS**2 / (18.0 * MU * dL)
    eps = (max(St, 0.0) / (St + 0.65)) ** 3.7
    return n12 * Ug / w * eps

def run_mednikov_size(models=None):
    """Mednikov K(d2); DEM and FF-PINN (SPGF+Stokes) at standing-wave antinode."""
    f, spl, d1, rho = float(FREQ_PINN), 150.0, 10e-6, 1000.0
    if models is None:
        models = load_models()

    # Standing-wave peak |u| (project x=0) = travelling Ug / sqrt(2)
    Ug_stand = u_travelling_peak(spl) / np.sqrt(2.0)
    x0 = 0.0

    d2_line = np.unique(np.concatenate([
        np.logspace(np.log10(0.3e-6), np.log10(60e-6), 200),
        np.linspace(0.5 * d1, 1.5 * d1, 160),
    ]))
    K_lit = np.array([mednikov_K(d1, d, f, spl, rho, Ug=Ug_stand) for d in d2_line])
    # True zero at d1=d2; floor only for log-axis visibility of the cusp
    K_lit_plot = np.maximum(K_lit, 1e-16)

    d2_pts = np.array([0.5, 1, 2, 5, 8, 12, 20, 40]) * 1e-6
    K_dem, K_pinn = [], []
    for d in d2_pts:
        K_dem.append(integrate_pair(
            d1, float(d), f, spl, rho, mode="project_dem", x0=x0, include_spgf=True,
        )[0])
        K_pinn.append(integrate_pair(
            d1, float(d), f, spl, rho, mode="pinn", x0=x0, include_spgf=True,
            pinn_models=models,
        )[0])
    K_dem = np.array(K_dem)
    K_pinn = np.array(K_pinn)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(d2_line * 1e6, K_lit_plot * 1e6, "k-", lw=2,
              label=r"Mednikov $\beta_{\mathrm{OI}}$ (antinode $U_g$)", zorder=1)
    ax.loglog(d2_pts * 1e6, np.maximum(K_dem, 1e-16) * 1e6, "o", ms=8, mfc="none", mec="C0", mew=1.7,
              label="DEM (SPGF + Stokes)", zorder=2)
    ax.loglog(d2_pts * 1e6, np.maximum(K_pinn, 1e-16) * 1e6, "+", ms=8, mec="C1", mew=1.7,
              label="FF-PINN (SPGF + Stokes)", zorder=3)
    ax.axvline(10.0, color="0.5", ls=":", lw=1)
    ax.set_xlabel(r"Partner diameter $d_2$ ($\mu$m)  [$d_1=10\,\mu$m]")
    ax.set_ylabel(r"Orthokinetic kernel $K$ (cm$^3$/s)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.3, 60)
    ax.set_ylim(3e-7, 1e-2)
    fig.tight_layout()
    png = OUT / "figA_compare_mednikov_kernel.png"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(OUT / "figA_compare_mednikov_kernel.pdf", facecolor="white")
    plt.close(fig)

    K_ref = np.array([mednikov_K(d1, d, f, spl, rho, Ug=Ug_stand) for d in d2_pts])
    mask = K_ref > 1e-16
    ratio_dem = float(np.median(K_dem[mask] / K_ref[mask]))
    ratio_pinn = float(np.median(K_pinn[mask] / K_dem[mask]))
    print(f"[A] DEM/Mednikov={ratio_dem:.3f}; PINN/DEM={ratio_pinn:.3f}", flush=True)
    return {
        "paper": "Mednikov orthokinetic K (Shang et al. 2018 form)",
        "conditions": {"f_Hz": f, "SPL_dB": spl, "rho_p": rho, "d1_um": 10.0, "wave": "standing antinode"},
        "dem_over_mednikov": ratio_dem,
        "pinn_over_dem": ratio_pinn,
        "figure": png.name,
    }


# =============================================================================
# B — Dong 2006 frequency
# =============================================================================

def run_dong2006_freq(models=None):
    """
    Dong (2006) Fig.4-style L_orth^eff(f) at SPL=136 dB, rho=2400.

    Triangles = trained FF-PINN packs (PINN/freq_*k) → entrainment → Dong Eqs.5–10.
    Available packs start at 2 kHz, so the axis is extended past Dong's original
    5 kHz cutoff to show those trained frequencies.
    """
    spl, rho = 136.0, 2400.0
    # Dong curve over original band + high-freq tail for PINN packs
    freqs = np.unique(np.concatenate([
        np.logspace(np.log10(40), np.log10(12000), 90),
        np.linspace(80, 300, 40),
    ]))
    pairs = [(10e-6, 6e-6, "C0", 140.0), (10e-6, 4e-6, "C1", 178.0)]

    f_dem = np.array([50.0, 90.0, 140.0, 180.0, 350.0, 800.0, 2000.0, 4000.0, 8000.0, 10000.0])
    # Prefer Dong fill-in Hz packs + existing kHz packs
    f_pinn_candidates = [
        90.0, 140.0, 180.0, 350.0, 800.0,
        2000.0, 4000.0, 6000.0, 8000.0, 10000.0,
    ]
    f_pinn = np.array([
        f for f in f_pinn_candidates
        if pack_dir_for_freq(f).joinpath("stokes_model_x.pth").exists()
    ])

    def _nearest(f_targets):
        idx = [int(np.argmin(np.abs(freqs - f))) for f in f_targets]
        out = []
        for i in idx:
            if not out or i != out[-1]:
                out.append(i)
        return np.array(out)

    mk_dem = _nearest(f_dem)

    # Cache PINN relaxation times (phase-lag; stable near full entrainment)
    need_d = sorted({10e-6, 6e-6, 4e-6})
    tau_pinn = {}  # (f_hz, d) -> tau
    print("  [B] loading trained PINN packs for frequency markers …", flush=True)
    for f in f_pinn:
        models_f = load_models_at_freq(float(f))
        print(f"       pack {models_f['pack_dir']} @ {f:.0f} Hz", flush=True)
        for d in need_d:
            tau = relaxation_time_from_trajectory(
                float(d), float(f), spl, rho,
                mode="pinn_dong", pinn_models=models_f, include_spgf=False, x_pinn=0.0,
            )
            tau_th = float(tau_p(d, rho))
            tau_pinn[(float(f), float(d))] = tau
            print(
                f"         d={d*1e6:.0f} um  tau_PINN={tau:.3e}  tau_th={tau_th:.3e}  "
                f"ratio={tau/max(tau_th,1e-30):.3f}",
                flush=True,
            )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    peaks = {}
    pinn_stats = []
    for d1, d2, c, f_paper in pairs:
        lab = rf"({d1*1e6:.0f},{d2*1e6:.0f}) $\mu$m"
        L_lit = np.array([dong_Leff(d1, d2, f, spl, rho) for f in freqs])
        L_dem = np.array([dong_Leff_dem(d1, d2, f, spl, rho, mode="stokes") for f in freqs])
        y_lit, y_dem = L_lit * 1e6, L_dem * 1e6

        L_pinn_pts = []
        for f in f_pinn:
            L_pinn_pts.append(_L_orth_from_tau(
                d1, d2,
                tau_pinn[(float(f), float(d1))],
                tau_pinn[(float(f), float(d2))],
                float(f), spl, rho,
            ))
        L_pinn_pts = np.asarray(L_pinn_pts)
        y_pinn = L_pinn_pts * 1e6
        L_lit_at_pinn = np.array([dong_Leff(d1, d2, f, spl, rho) for f in f_pinn])
        ratio = float(np.median(L_pinn_pts / np.maximum(L_lit_at_pinn, 1e-30)))
        pinn_stats.append({"pair": lab, "pinn_over_dong_median": ratio})

        ax.semilogx(
            freqs, y_lit, "-", color=c, lw=2.0,
            label=rf"Dong et al. analytical results {lab}", zorder=1,
        )
        ax.semilogx(
            freqs[mk_dem], y_dem[mk_dem], "s", color=c, ms=6.5, mfc=c, mew=0.6,
            alpha=0.9, label=rf"DEM {lab}", zorder=4,
        )
        ax.semilogx(
            f_pinn, y_pinn, "^", color=c, ms=7.5, mfc="w", mew=1.6,
            label=rf"FF-PINN (trained) {lab}", zorder=5,
        )

        i = int(np.argmax(L_lit))
        peaks[lab] = {
            "lit_f_Hz": float(freqs[i]),
            "lit_L_um": float(L_lit[i] * 1e6),
            "paper_f_opt_Hz": f_paper,
            "dem_f_Hz": float(freqs[int(np.argmax(L_dem))]),
            "dem_L_um": float(L_dem[int(np.argmax(L_dem))] * 1e6),
            "pinn_freqs_Hz": f_pinn.tolist(),
            "pinn_L_um": y_pinn.tolist(),
            "pinn_over_dong_median": ratio,
        }

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Orthokinetic $L_{\mathrm{orth}}^{\mathrm{eff}}$ ($\mu$m)")
    ax.set_xlim(40, 12000)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    png = OUT / "figA_compare_dong2006_frequency.png"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(OUT / "figA_compare_dong2006_frequency.pdf", facecolor="white")
    plt.close(fig)
    print(f"[B] peaks: {peaks}", flush=True)
    print(f"[B] pinn_stats: {pinn_stats}", flush=True)
    return {
        "paper": "Dong et al. (2006) Fig.4 (axis extended for trained PINN packs)",
        "pinn_weights_used": True,
        "pinn_freqs_Hz": f_pinn.tolist(),
        "marker_meaning": "FF-PINN trained packs → entrainment → Dong Eqs.5–10",
        "peaks": peaks,
        "pinn_stats": pinn_stats,
        "figure": png.name,
    }


# =============================================================================
# C — González 2000 + PINN @ 10 kHz
# =============================================================================

def run_gonzalez_pinn():
    d_g, rho_g = 7.9e-6, 2400.0
    freqs = np.logspace(np.log10(20), np.log10(10000), 100)
    eta_th = np.array([eta_bfh(d_g, f, rho_g) for f in freqs])

    models = load_models()
    diams = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    eta_pinn, eta_bfh_10k = [], []
    for du in diams:
        d = du * 1e-6
        rng = np.random.default_rng(int(du * 100))
        n, x, vx = 100, rng.uniform(0, DOMAIN, 100), np.zeros(100)
        mass = RHO_PINN * np.pi * d**3 / 6
        dt, steps = 1e-7, 10000
        vpk = np.zeros(n)
        for step in range(steps):
            t = step * dt
            F = force_for_diameter(models, x, vx, t, d)
            vx = vx + (F / mass) * dt
            x = np.mod(x + vx * dt, DOMAIN)
            if step >= steps - 1500:
                vpk = np.maximum(vpk, np.abs(vx))
        ugpk = np.zeros(n)
        for t in np.linspace(0, 1 / FREQ_PINN, 30, endpoint=False):
            _, u = pinn_fields(models, x, t)
            ugpk = np.maximum(ugpk, np.abs(u))
        good = ugpk > 0.35 * u0_amplitude()
        eta_pinn.append(float(np.median(vpk[good] / ugpk[good])))
        eta_bfh_10k.append(float(eta_bfh(d, FREQ_PINN, RHO_PINN)))
        print(f"  [C] d={du} PINN={eta_pinn[-1]:.3f} BFH={eta_bfh_10k[-1]:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.semilogx(freqs, eta_th, "k-", lw=2,
                label=r"BFH $\eta=[1+(\omega\tau)^2]^{-1/2}$")
    a1.axvspan(20, 3500, color="0.85", alpha=0.5, label="González (2000) exp. window")
    a1.set_xlabel("Frequency (Hz)")
    a1.set_ylabel(r"Entrainment $\eta$")
    a1.legend(fontsize=7.5)
    a1.grid(True, which="both", alpha=0.3)

    a2.plot(diams, eta_bfh_10k, "k-", lw=2, label="BFH @ 10 kHz")
    a2.plot(diams, eta_pinn, "ro", ms=8, label="FF-PINN")
    a2.set_xlabel(r"Diameter $d$ ($\mu$m)")
    a2.set_ylabel(r"Entrainment $\eta$")
    a2.legend(fontsize=8.5)
    a2.grid(True, alpha=0.3)

    fig.tight_layout()
    png = OUT / "figA_compare_gonzalez2000_entrainment.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    rel = max(abs(a - b) / b for a, b in zip(eta_pinn, eta_bfh_10k))
    return {"paper": "Gonzalez (2000)", "pinn_used": True, "max_rel_err": rel, "figure": png.name}


# =============================================================================
# D — Dong 2006 size @ 10 kHz (PINN applicable)
# =============================================================================

def run_dong2006_size(models):
    """
    Dong (2006) orthokinetic-only comparison at Fig.5b acoustic conditions
    (f=10 kHz, SPL=160 dB, rho=2400). Uses Dong Eqs.(5)–(10) for L_orth^eff
    — the paper's wake-free orthokinetic branch. Wake is not plotted.
    """
    f, spl, rho = 10000.0, 160.0, 2400.0
    d_small = np.linspace(0.15, 5.95, 120) * 1e-6
    d_pts = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-6
    dLs = [4e-6, 6e-6]

    # Cache PINN η once per diameter (reuse across d_L pairs)
    need = sorted({float(d) for d in list(d_pts) + dLs})
    eta_pinn = {}
    print("  [D] caching PINN entrainment …", flush=True)
    for d in need:
        eta_pinn[d] = entrainment_single(
            d, f, spl, rho, mode="pinn", pinn_models=models, include_spgf=False
        )
        print(f"       d={d*1e6:.1f} um  eta_PINN={eta_pinn[d]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    stats = []

    for dL, c in zip(dLs, ["C0", "C1"]):
        L_orth = np.array([dong_Leff_ortho(dL, d, f, spl, rho) for d in d_small])
        ax.plot(d_small * 1e6, L_orth * 1e6, "-", color=c, lw=2.4,
                label=rf"Dong $L_{{\mathrm{{orth}}}}^\mathrm{{eff}}$ (Eqs.5–10), $d_L={dL*1e6:.0f}\,\mu$m")

        L_dem, L_pinn = [], []
        for d in d_pts:
            d = float(d)
            L_dem.append(dong_Leff_dem(dL, d, f, spl, rho, mode="stokes"))
            L_pinn.append(_L_orth_from_eta(dL, d, eta_pinn[dL], eta_pinn[d], f, spl, rho))

        L_dem = np.array(L_dem)
        L_pinn = np.array(L_pinn)
        L_lit_pts = np.array([dong_Leff_ortho(dL, d, f, spl, rho) for d in d_pts])
        mask = L_lit_pts > 1e-9
        r_dem = float(np.median(L_dem[mask] / L_lit_pts[mask])) if np.any(mask) else np.nan
        r_pin = float(np.median(L_pinn[mask] / L_dem[mask])) if np.any(mask) else np.nan
        stats.append({"dL_um": dL * 1e6, "dem_over_lit_ortho": r_dem, "pinn_over_dem": r_pin})
        print(f"  [D] d_L={dL*1e6:.0f}  DEM/L_orth={r_dem:.3f}  PINN/DEM={r_pin:.3f}", flush=True)

        ax.plot(d_pts * 1e6, L_dem * 1e6, "o", color=c, ms=7, mfc="w", mew=1.6,
                label=rf"DEM $\eta\to$ Dong, $d_L={dL*1e6:.0f}\,\mu$m")
        ax.plot(d_pts * 1e6, L_pinn * 1e6, "s", color=c, ms=5.5, alpha=0.9,
                label=rf"FF-PINN $\eta\to$ Dong, $d_L={dL*1e6:.0f}\,\mu$m")

    ax.set_xlabel(r"Small-particle diameter $d_S$ ($\mu$m)")
    ax.set_ylabel(r"Orthokinetic $L_{\mathrm{orth}}^{\mathrm{eff}}$ ($\mu$m)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.2, 6.0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    png = OUT / "figA_compare_dong2006_size.png"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "paper": "Dong (2006) Eqs.5–10 orthokinetic L_eff (Fig.5b conditions)",
        "conditions": {"f_Hz": f, "SPL_dB": spl, "rho_p": rho},
        "quantity": "L_orth^eff (wake-free)",
        "stats": stats,
        "figure": png.name,
        "note": "Compares Dong's orthokinetic formula (Eqs.5–10); wake not plotted.",
    }


def main():
    print("=== Appendix A literature comparison ===", flush=True)
    models = load_models()
    A = run_mednikov_size(models)
    B = run_dong2006_freq(models)
    C = run_gonzalez_pinn()
    D = run_dong2006_size(models)
    audit = {
        "notes": [
            "Literature curves: closed-form equations from the cited papers.",
            "DEM: mechanisms.Stokes_drag (per-particle d) + optional mechanisms.ARF SPGF.",
            "FF-PINN: _kernel_common.force_for_diameter (10 kHz weights).",
            "Wake effect omitted (outside model scope).",
        ],
        "A_mednikov": A, "B_dong2006_freq": B,
        "C_gonzalez2000": C, "D_dong2006_size": D,
    }
    (DATA / "literature_kernel_comparison.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("WROTE literature_kernel_comparison.json")


if __name__ == "__main__":
    main()
