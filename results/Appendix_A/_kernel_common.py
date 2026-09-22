"""
Shared physics for Appendix A orthokinetic-kernel analysis.

Key fix vs. the earlier scripts: the shipped PINN force is frozen at the training
diameter d_ref = 2 um (drag coefficient and SPGF magnitude both fixed at 2 um), so
using only mass ~ d^3 to rescale gives an *effective* relaxation time tau ~ d^3,
i.e. a WRONG diameter dependence of the entrainment factor.

Here we keep the PINN's learned, diameter-INDEPENDENT gas-velocity field u_air and
the SPGF spatial/temporal shape, but apply the correct diameter scaling:
    - Stokes drag coefficient  c(d)   = 3*pi*mu*d / Cc(d)          (linear in d)
    - Cunningham slip          Cc(d)  = 1 + Kn[1.257 + 0.4 exp(-1.1/Kn)], Kn = 2*lam/d
    - SPGF magnitude           ~ (d/d_ref)^2                        (projected area)
so that tau(d) = m(d)/c(d) = rho_p Cc d^2 / (18 mu) ~ d^2, matching orthokinetic theory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mechanisms.ARF_PINN_t import ARFNetT
from mechanisms.ARF_PINN_x import ARFNet as ARFNetX
from mechanisms.STOKES_PINN_t import StokesNetT
from mechanisms.STOKES_PINN_v import StokesNetV
from mechanisms.STOKES_PINN_x import StokesNetX
from mechanisms.UNIFIED_PINN import UnifiedNormalizer
from initialization.sound_source_standing import frequency, sound_pressure_level

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
M_CKPT = 32
FREQ = float(frequency)
MU = 1.86e-5           # match UNIFIED_PINN closure
RHO_P = 2000.0
RHO0 = 1.225
C0 = 340.0
PREF = 20e-6
LAMBDA_G = 6.5e-8      # mean free path -> Cc(2um)=1.0817 (matches shipped closure)
D_REF = 2e-6
CC_REF = 1.0817
DOMAIN = 0.034


def cunningham(d):
    kn = 2.0 * LAMBDA_G / d
    return 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))


def relaxation_time(d):
    return RHO_P * d**2 * cunningham(d) / (18.0 * MU)


def entrainment(d):
    """Classical amplitude ratio eta = 1/sqrt(1+(omega tau)^2)."""
    omega = 2 * np.pi * FREQ
    return 1.0 / np.sqrt(1.0 + (omega * relaxation_time(d)) ** 2)


def u0_amplitude():
    omega = 2 * np.pi * FREQ
    p = PREF * (10 ** (sound_pressure_level / 20.0))
    A = p / (RHO0 * C0 * omega)
    return omega * A


def pack_dir_for_freq(freq_hz: float) -> Path:
    """Return a trained pack dir for ``freq_hz``.

    Prefer ``PINN/freq_<Hz>Hz`` (Dong fill-in packs), then ``PINN/freq_<k>k``,
    then root ``PINN/`` for the production 10 kHz checkpoint.
    """
    hz = int(round(float(freq_hz)))
    hz_pack = ROOT / "PINN" / f"freq_{hz}Hz"
    if (hz_pack / "stokes_model_x.pth").exists():
        return hz_pack
    if hz % 1000 == 0:
        khz = hz // 1000
        dedicated = ROOT / "PINN" / f"freq_{khz}k"
        if (dedicated / "stokes_model_x.pth").exists():
            return dedicated
    if abs(float(freq_hz) - 10000.0) < 1.0:
        return ROOT / "PINN"
    # fallback guess for non-integer-kHz without Hz pack
    khz = max(1, int(round(float(freq_hz) / 1000.0)))
    return ROOT / "PINN" / f"freq_{khz}k"


def load_models(pack_dir: Path | None = None, freq_hz: float | None = None):
    """Load a FF-PINN checkpoint pack. Defaults to root PINN/ at project frequency."""
    f_hz = float(FREQ if freq_hz is None else freq_hz)
    base = Path(pack_dir) if pack_dir is not None else ROOT / "PINN"
    models = {
        "arf_x": ARFNetX(fourier_features=M_CKPT).to(DEVICE),
        "arf_t": ARFNetT(period_seconds=1.0 / f_hz).to(DEVICE),
        "stokes_x": StokesNetX(fourier_features=M_CKPT).to(DEVICE),
        "stokes_t": StokesNetT(period_seconds=1.0 / f_hz).to(DEVICE),
        "stokes_v": StokesNetV().to(DEVICE),
    }
    for name, fn in [
        ("arf_x", "arf_model_x.pth"), ("arf_t", "arf_model_t.pth"),
        ("stokes_x", "stokes_model_x.pth"), ("stokes_t", "stokes_model_t.pth"),
        ("stokes_v", "stokes_model_v.pth"),
    ]:
        models[name].load_state_dict(torch.load(base / fn, map_location=DEVICE, weights_only=True))
        models[name].eval()
    un = UnifiedNormalizer()
    un.load_from_individual_models(
        {
            "arf_x_norm": str(base / "arf_model_x_normalization_params.json"),
            "arf_t_norm": str(base / "arf_model_t_normalization_params.json"),
            "stokes_x_norm": str(base / "stokes_model_x_normalization_params.json"),
            "stokes_t_norm": str(base / "stokes_model_t_normalization_params.json"),
            "stokes_v_norm": str(base / "stokes_model_v_normalization_params.json"),
        },
        DEVICE,
    )
    models["unified_norm"] = un
    models["freq_hz"] = f_hz
    models["pack_dir"] = str(base)
    return models


def load_models_at_freq(freq_hz: float):
    """Load the trained pack for ``freq_hz`` (Hz), e.g. 2000 → PINN/freq_2k."""
    return load_models(pack_dir=pack_dir_for_freq(freq_hz), freq_hz=freq_hz)


@torch.no_grad()
def pinn_fields(models, x_np, t, spl=None):
    """Return (SPGF at d_ref [N], gas velocity u_air [N]) from the PINN at positions x, time t.

    ``spl`` overrides the project default sound_pressure_level so the same
    trained shape factors can be evaluated at a paper's SPL (amplitude scales
    with p/(rho c); no re-training).
    """
    un = models["unified_norm"]
    f_hz = float(models.get("freq_hz", FREQ))
    x = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    vx = torch.zeros_like(x)
    tt = torch.full_like(x, float(t))

    x_norm, _, t_norm = un.normalize_inputs(x, vx, tt)
    arf_fx_norm = models["arf_x"](x_norm.unsqueeze(1))
    arf_ft_norm = models["arf_t"](t_norm.unsqueeze(1))
    arf_fx, arf_ft = un.denormalize_arf_force(arf_fx_norm, arf_ft_norm)
    spgf_ref = (arf_fx.squeeze() * arf_ft.squeeze()).cpu().numpy()

    sx, _, st = un.normalize_inputs_stokes(x, vx, tt)
    stokes_fx = models["stokes_x"](sx.unsqueeze(1))
    stokes_ft = models["stokes_t"](st.unsqueeze(1))
    fx, ft, _ = un.denormalize_stokes_force(stokes_fx, stokes_ft, stokes_fx)
    omega = 2.0 * np.pi * f_hz
    spl_use = float(sound_pressure_level if spl is None else spl)
    p = PREF * (10 ** (spl_use / 20.0))
    A = p / (RHO0 * C0 * omega)
    # SPGF training used project SPL; rescale projected-area force with (p/p_train)^2
    # because pressure ~ p and SPGF ~ Δp ~ p (actually ~p for linear standing wave).
    # Linear acoustic pressure ~ SPL amplitude -> SPGF ~ p. Scale once.
    p_train = PREF * (10 ** (float(sound_pressure_level) / 20.0))
    spgf_ref = spgf_ref * (p / p_train)
    u_air = (-omega * A * fx.squeeze() * ft.squeeze()).cpu().numpy()
    return spgf_ref, u_air


def force_for_diameter(models, x_np, vx_np, t, d, spl=None, include_spgf=True):
    """
    Diameter-correct force (N) on size-d particles at positions x, velocities vx, time t.
    Uses PINN gas-velocity field u_air (diameter independent) and SPGF shape, applying
    physical diameter scaling to drag coefficient and SPGF magnitude.

    Set ``include_spgf=False`` for orthokinetic-only comparisons (Mednikov).
    """
    spgf_ref, u_air = pinn_fields(models, x_np, t, spl=spl)
    drag_coeff = 3.0 * np.pi * MU * d / cunningham(d)       # linear in d
    stokes = -drag_coeff * (vx_np - u_air)
    if not include_spgf:
        return stokes
    spgf = spgf_ref * (d / D_REF) ** 2                     # SPGF ~ projected area ~ d^2
    return spgf + stokes
