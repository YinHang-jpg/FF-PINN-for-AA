"""Generate 10 kHz FF-PINN trajectories for t = 0 .. 1 ms (1 us steps, save every 5 us)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from initialization.particle_initialization import initialize_particles
from mechanisms.UNIFIED_PINN import (
    PhysicalUnifiedModel,
    UnifiedNormalizer,
    load_individual_models,
)

FREQUENCY_HZ = 10000.0
DT = 1e-6
T_MAX = 1e-3
SAVE_EVERY = 5e-6
N_PARTICLES = 4400
OUT_DIR = PROJECT_ROOT / "front_page" / "anim_data"


def load_pinn(device):
    model_base = PROJECT_ROOT / "PINN"
    models = load_individual_models(device, str(model_base))
    unified_norm = UnifiedNormalizer()
    model_paths = {
        "arf_x_norm": str(model_base / "arf_model_x_normalization_params.json"),
        "arf_t_norm": str(model_base / "arf_model_t_normalization_params.json"),
        "stokes_x_norm": str(model_base / "stokes_model_x_normalization_params.json"),
        "stokes_t_norm": str(model_base / "stokes_model_t_normalization_params.json"),
        "stokes_v_norm": str(model_base / "stokes_model_v_normalization_params.json"),
    }
    unified_norm.load_from_individual_models(model_paths, device="cpu")
    return PhysicalUnifiedModel(
        models, unified_norm, device=device, frequency=FREQUENCY_HZ
    )


def integrate_pinn(model, pos0, vel0, mass, device):
    steps = int(round(T_MAX / DT))
    stride = max(1, int(round(SAVE_EVERY / DT)))
    pos = torch.as_tensor(pos0, dtype=torch.float32, device=device)
    vel = torch.as_tensor(vel0, dtype=torch.float32, device=device)
    inv_mass = 1.0 / torch.as_tensor(mass, dtype=torch.float32, device=device)
    times, xs = [], []
    t = 0.0
    with torch.inference_mode():
        for step in range(steps + 1):
            if step % stride == 0:
                times.append(t)
                xs.append(pos[:, 0].detach().cpu().numpy() * 1000.0)
            if step == steps:
                break
            x = pos[:, 0]
            vx = vel[:, 0]
            t_vec = torch.full_like(x, float(t))
            fx = model(x, vx, t_vec)
            pos = pos.clone()
            vel = vel.clone()
            pos[:, 0] = pos[:, 0] + vel[:, 0] * DT
            pos[:, 1] = pos[:, 1] + vel[:, 1] * DT
            vel[:, 0] = vel[:, 0] + fx * inv_mass * DT
            t += DT
    return np.asarray(times), np.asarray(xs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pos0, vel0, _, mass = initialize_particles(N=N_PARTICLES, domain_size=(0.034, 0.034))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PINN integrate on {device}: t=0..{T_MAX * 1e3:.3f} ms, dt={DT * 1e6:.0f} us")
    model = load_pinn(device)
    times, x_mm = integrate_pinn(model, pos0, vel0, mass, device)
    out = OUT_DIR / "pinn_early_1ms.csv"
    np.savetxt(out, np.column_stack([times, x_mm]), delimiter=",")
    print(f"wrote {out} ({len(times)} frames, {x_mm.shape[1]} particles)")


if __name__ == "__main__":
    main()
