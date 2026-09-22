"""
Co-located two-particle trajectory kernels using project physics modules.

Modes
-----
* ``stokes``  — orthokinetic fair-compare: per-particle Stokes drag + travelling-wave
              gas velocity (``mechanisms.Stokes_drag`` closure). Matches Mednikov
              orthokinetic kernel definition (no wake, no ARF).
* ``dem``     — travelling-wave Stokes + standing-wave SPGF (legacy literature sweeps).
* ``project_dem`` — **project DEM**: standing-wave gas velocity
              (``mechanisms.Stokes_drag.compute_air_velocity_due_to_sound``) + per-particle
              Stokes + SPGF/ARF at the same SPL/f as the PINN training setup.
* ``pinn``    — FF-PINN unified force via ``_kernel_common.force_for_diameter`` (10 kHz
              weights only; must share SPL / wave type / rho_p with ``project_dem``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mechanisms.Stokes_drag import (  # noqa: E402
    cunningham_correction_factor,
    compute_air_velocity_due_to_sound,
)

MU = 1.86e-5
RHO0 = 1.225
C0 = 340.0
PREF = 20e-6
LAMBDA_G = 6.5e-8
P0 = 101325.0
GAMMA = 1.4


def u_travelling_peak(spl_db: float) -> float:
    """
    Peak gas velocity amplitude Ug for a plane travelling wave from SPL.

    Dong et al. (2006): SPL = 20 log10(prms / 20e-6), prms = Pg/√2,
    Pg = ρ0 c0 Ug  ⇒  Ug = prms * √2 / (ρ0 c0).
    (Missing √2 under-predicts Ug by √2; paper: Ug≈2.7 m/s at 152 dB.)
    """
    prms = PREF * (10 ** (spl_db / 20.0))
    return prms * np.sqrt(2.0) / (RHO0 * C0)


def u_travelling(t: float, f: float, spl_db: float) -> float:
    Ug = u_travelling_peak(spl_db)
    w = 2 * np.pi * f
    return Ug * np.cos(w * t)


def spgf_x(x: float, d: float, t: float, f: float, spl_db: float) -> float:
    """ARF / SPGF x-force (N) — same standing-wave pressure difference as mechanisms.ARF."""
    p = PREF * (10 ** (spl_db / 20.0))
    A = p / (RHO0 * C0 * 2 * np.pi * f)
    lam = C0 / f
    k = 2 * np.pi / lam
    w = 2 * np.pi * f
    off = np.sqrt(2) * d / 4
    def p_at(xq):
        return (
            2 * 2 * np.pi * A * P0 * GAMMA
            * np.sin(k * xq) * np.cos(w * t) / lam
        )
    return np.pi * d**2 * (p_at(x - off) - p_at(x + off)) / 4


def stokes_fx(vx: float, d: float, u_gas: float) -> float:
    """Dong/Mednikov Stokes drag (Eq.1): F = -3πμd (v - u); no Cunningham slip."""
    c = 3 * np.pi * MU * d
    return -c * (vx - u_gas)


def integrate_pair(
    d1: float,
    d2: float,
    f: float,
    spl_db: float,
    rho_p: float,
    mode: str = "stokes",
    x0: float = 0.0,
    n_cycles: float = 6.0,
    pinn_models=None,
    include_spgf: bool = True,
) -> tuple[float, float, float]:
    """
    Integrate two co-located particles; return (K, dv_amp, Ug_peak).

    K uses the Mednikov relative-velocity *amplitude* estimated from the
    cycle mean:  dv_amp = < |v1-v2| > * (pi/2)  for near-sinusoidal motion.
    This matches K = (pi/4)(d1+d2)^2 |eta1-eta2| Ug, not a peak-value overestimate.
    """
    m1 = rho_p * np.pi * d1**3 / 6
    m2 = rho_p * np.pi * d2**3 / 6
    T = 1.0 / f
    tau_min = min(rho_p * d1**2 / (18 * MU), rho_p * d2**2 / (18 * MU))
    dt = min(T / 200.0, 0.15 * tau_min)
    n = int(np.ceil(n_cycles * T / dt))
    v1 = v2 = 0.0
    n0 = n // 2
    pos1 = np.array([[x0, 0.0]])
    dv_sum = 0.0
    n_meas = 0

    if mode in ("standing", "project_dem", "pinn"):
        ug_peak = 0.0
        for ti in np.linspace(0, T, 80, endpoint=False):
            ug_peak = max(
                ug_peak,
                abs(float(compute_air_velocity_due_to_sound(pos1, ti, f, spl_db)[0, 0])),
            )
        Ug = ug_peak
    else:
        Ug = u_travelling_peak(spl_db)

    for i in range(n):
        t = i * dt
        if mode in ("standing", "project_dem", "pinn"):
            u = float(compute_air_velocity_due_to_sound(pos1, t, f, spl_db)[0, 0])
        else:
            u = u_travelling(t, f, spl_db)
        if mode == "stokes":
            f1 = stokes_fx(v1, d1, u)
            f2 = stokes_fx(v2, d2, u)
        elif mode == "dem":
            f1 = stokes_fx(v1, d1, u) + (spgf_x(x0, d1, t, f, spl_db) if include_spgf else 0.0)
            f2 = stokes_fx(v2, d2, u) + (spgf_x(x0, d2, t, f, spl_db) if include_spgf else 0.0)
        elif mode in ("standing", "project_dem"):
            f1 = stokes_fx(v1, d1, u) + (spgf_x(x0, d1, t, f, spl_db) if include_spgf else 0.0)
            f2 = stokes_fx(v2, d2, u) + (spgf_x(x0, d2, t, f, spl_db) if include_spgf else 0.0)
        elif mode == "pinn":
            if pinn_models is None:
                raise ValueError("pinn mode requires loaded models")
            from _kernel_common import force_for_diameter
            f1 = float(force_for_diameter(
                pinn_models, np.array([x0]), np.array([v1]), t, d1,
                spl=spl_db, include_spgf=include_spgf,
            )[0])
            f2 = float(force_for_diameter(
                pinn_models, np.array([x0]), np.array([v2]), t, d2,
                spl=spl_db, include_spgf=include_spgf,
            )[0])
        else:
            raise ValueError(mode)
        v1 += (f1 / m1) * dt
        v2 += (f2 / m2) * dt
        if i >= n0:
            dv_sum += abs(v1 - v2)
            n_meas += 1

    dv_amp = (dv_sum / max(n_meas, 1)) * (np.pi / 2.0)  # mean(|sin|) = 2/pi
    K = (np.pi / 4.0) * (d1 + d2) ** 2 * dv_amp
    return K, dv_amp, Ug


def _drive_and_force(d, f, spl_db, mode, t, v, x_pinn, include_spgf, pinn_models):
    """Return (u_gas, force) at one time for the chosen mode."""
    if mode in ("standing", "project_dem"):
        u = float(compute_air_velocity_due_to_sound(np.array([[0.0, 0.0]]), t, f, spl_db)[0, 0])
    elif mode in ("pinn", "pinn_dong"):
        from _kernel_common import pinn_fields
        _, u_arr = pinn_fields(pinn_models, np.array([x_pinn]), t, spl=spl_db)
        u = float(np.asarray(u_arr).reshape(-1)[0])
    else:
        u = u_travelling(t, f, spl_db)

    if mode == "stokes":
        fx = stokes_fx(v, d, u)
    elif mode == "pinn_ortho":
        from _kernel_common import cunningham, MU as MU_P
        drag = 3.0 * np.pi * MU_P * d / cunningham(d)
        fx = -drag * (v - u)
    elif mode == "pinn_dong":
        # PINN gas field + Dong Stokes drag (no Cunningham) for literature L_orth
        fx = stokes_fx(v, d, u)
    elif mode in ("dem", "standing", "project_dem"):
        fx = stokes_fx(v, d, u) + (spgf_x(0.0, d, t, f, spl_db) if include_spgf else 0.0)
    elif mode == "pinn":
        from _kernel_common import force_for_diameter
        fx = float(force_for_diameter(
            pinn_models, np.array([x_pinn]), np.array([v]), t, d,
            spl=spl_db, include_spgf=include_spgf,
        )[0])
    else:
        raise ValueError(mode)
    return u, fx


def entrainment_single(d: float, f: float, spl_db: float, rho_p: float, mode: str = "stokes", **kw) -> float:
    """eta = peak|vp| / peak|ug| for one particle (same driving field in num/denom)."""
    include_spgf = kw.get("include_spgf", True)
    m = rho_p * np.pi * d**3 / 6
    T = 1.0 / f
    tau = rho_p * d**2 / (18 * MU)
    dt = min(T / 200.0, 0.15 * tau)
    n = int(np.ceil(8 * T / dt))
    v = 0.0
    vpk = 0.0
    ugpk = 0.0
    x_pinn = kw.get("x_pinn", None)
    if mode in ("pinn", "pinn_dong") and x_pinn is None:
        x_pinn = 0.0
    for i in range(n):
        t = i * dt
        u, fx = _drive_and_force(
            d, f, spl_db, mode, t, v, x_pinn, include_spgf, kw.get("pinn_models"),
        )
        ugpk = max(ugpk, abs(u))
        v += (fx / m) * dt
        if i >= n // 2:
            vpk = max(vpk, abs(v))
    return vpk / (ugpk + 1e-30)


def relaxation_time_from_trajectory(
    d: float, f: float, spl_db: float, rho_p: float, mode: str = "stokes", **kw
) -> float:
    """
    Stokes relaxation time from fundamental phase lag (stable when eta → 1).

    For vp/ug = 1/(1 + i ωτ):  τ = Im(1/H) / ω, with H the complex transfer
    estimated over the last acoustic cycles. Prefer this over inverting
    eta = 1/sqrt(1+(ωτ)^2) near full entrainment.
    """
    include_spgf = kw.get("include_spgf", True)
    m = rho_p * np.pi * d**3 / 6
    T = 1.0 / f
    tau_guess = rho_p * d**2 / (18 * MU)
    dt = min(T / 200.0, 0.15 * max(tau_guess, 1e-9))
    n = int(np.ceil(10 * T / dt))
    n0 = n // 2
    v = 0.0
    x_pinn = kw.get("x_pinn", None)
    if mode in ("pinn", "pinn_dong") and x_pinn is None:
        x_pinn = 0.0
    w = 2.0 * np.pi * f
    # accumulate complex Fourier coeffs at ω
    Ug_c = 0.0 + 0.0j
    Vp_c = 0.0 + 0.0j
    n_meas = 0
    for i in range(n):
        t = i * dt
        u, fx = _drive_and_force(
            d, f, spl_db, mode, t, v, x_pinn, include_spgf, kw.get("pinn_models"),
        )
        v += (fx / m) * dt
        if i >= n0:
            z = np.exp(-1j * w * t)
            Ug_c += u * z
            Vp_c += v * z
            n_meas += 1
    if n_meas < 10 or abs(Ug_c) < 1e-30:
        # fallback amplitude route
        eta = entrainment_single(d, f, spl_db, rho_p, mode=mode, **kw)
        eta = float(np.clip(eta, 1e-12, 1.0 - 1e-15))
        return float(np.sqrt(max(1.0 / eta**2 - 1.0, 0.0)) / w)
    H = Vp_c / Ug_c
    # 1/H = 1 + i ωτ  ⇒  τ = Im(1/H)/ω
    invH = 1.0 / H
    tau = float(np.imag(invH) / w)
    return max(tau, 0.0)
