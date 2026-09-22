"""
Figure 7 — five FF-PINN sub-networks: loss (left) + theory vs PINN (right).

All five networks are trained and evaluated at the project carrier frequency
so theory curves share the same k / omega as the checkpoints.

Targets (same as mechanisms/*_PINN_*.py):
  ARF_x  1e-6 | ARF_t  1e-7 | Stokes_x/t/v  1e-8

Checkpoints: results/Fig7/data/fig7_ckpt/
Cache:       results/Fig7/data/fig7_training_losses.json
Figures:     figures/fig07_*.png

Usage:
  python results/Fig7/plot_fig7_training_losses.py
  python results/Fig7/plot_fig7_training_losses.py --retrain
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from _common import DATA, TRAINING_TARGETS, save_fig

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = DATA / "fig7_training_losses.json"
CKPT_DIR = DATA / "fig7_ckpt"
MAX_EPOCHS = 120_000
N_SAMPLES = 20_000
FOURIER_M = 32

THEORY_C = "#1f4e79"
PINN_C = "#c0392b"

SPECS = [
    ("ARF_x", "SPGF Spatial-Force PINN Training Loss", "SPGF position factor: theory vs PINN", TRAINING_TARGETS["ARF_x"]),
    ("ARF_t", "SPGF Time-Factor PINN Training Loss", "SPGF time factor vs time", TRAINING_TARGETS["ARF_t"]),
    ("Stokes_x", "Stokes Spatial-Dependence PINN Training Loss", "Stokes position factor: theory vs PINN", TRAINING_TARGETS["Stokes_x"]),
    ("Stokes_t", "Stokes Time-Dependence PINN Training Loss", "Stokes time factor: theory vs PINN", TRAINING_TARGETS["Stokes_t"]),
    ("Stokes_v", "Stokes Velocity-Dependence PINN Training Loss", "Stokes velocity factor: theory vs PINN", TRAINING_TARGETS["Stokes_v"]),
]


@dataclass
class TrainBundle:
    model: nn.Module
    meta: dict
    losses: list[float]


def _train_loop(model, x_in, y_tgt, target_loss: float) -> list[float]:
    opt = optim.Adam(model.parameters(), lr=5e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5000)
    crit = nn.MSELoss()
    losses: list[float] = []
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model(x_in), y_tgt)
        loss.backward()
        opt.step()
        sched.step(loss.item())
        lv = float(loss.item())
        losses.append(lv)
        if lv < target_loss:
            break
    return losses


def _build_and_train(key: str, target: float, device) -> TrainBundle:
    from mechanisms.ARF import c_0, frequency as f_hz
    from mechanisms.ARF_PINN_t import ARFNetT, theoretical_time_factor
    from mechanisms.ARF_PINN_x import ARFNet, theoretical_arf
    from mechanisms.STOKES_PINN_t import StokesNetT, theoretical_stokes_force_t
    from mechanisms.STOKES_PINN_v import StokesNetV, theoretical_stokes_force_v
    from mechanisms.STOKES_PINN_x import StokesNetX, sound_speed, theoretical_stokes_force_x

    period = 1.0 / float(f_hz)
    wavelength = float(c_0) / float(f_hz)
    meta: dict = {"key": key, "frequency_Hz": float(f_hz)}

    if key == "ARF_x":
        x_range = 3.0 * (wavelength / 2.0)
        x = (torch.rand(N_SAMPLES, 1, device=device) * x_range - x_range / 2.0).float()
        y = theoretical_arf(x).detach()
        x_min, x_max = -x_range / 2.0, x_range / 2.0
        x_in = (x - x_min) / (x_max - x_min)
        model = ARFNet(fourier_features=FOURIER_M).to(device)
        meta.update({"kind": "x_mm", "x_min": float(x_min), "x_max": float(x_max)})
    elif key == "ARF_t":
        t = (torch.rand(N_SAMPLES, 1, device=device) * period).float()
        y = theoretical_time_factor(t).detach()
        t_min, t_max = 0.0, period
        x_in = (t - t_min) / (t_max - t_min)
        model = ARFNetT(period_seconds=period).to(device)
        meta.update({"kind": "t_us", "t_min": t_min, "t_max": float(t_max)})
    elif key == "Stokes_x":
        x_range = 3.0 * (float(sound_speed) / float(f_hz))
        x = (torch.rand(N_SAMPLES, 1, device=device) * x_range - x_range / 2.0).float()
        y = theoretical_stokes_force_x(x).detach()
        x_min, x_max = -x_range / 2.0, x_range / 2.0
        x_in = (x - x_min) / (x_max - x_min)
        model = StokesNetX(fourier_features=FOURIER_M).to(device)
        meta.update({"kind": "x_mm", "x_min": float(x_min), "x_max": float(x_max)})
    elif key == "Stokes_t":
        t = (torch.rand(N_SAMPLES, 1, device=device) * period).float()
        y = theoretical_stokes_force_t(t).detach()
        t_min, t_max = 0.0, period
        x_in = (t - t_min) / (t_max - t_min)
        model = StokesNetT(period_seconds=period).to(device)
        meta.update({"kind": "t_us", "t_min": t_min, "t_max": float(t_max)})
    elif key == "Stokes_v":
        v_min, v_max = -0.1, 0.1
        v = (torch.rand(N_SAMPLES, 1, device=device) * (v_max - v_min) + v_min).float()
        y = theoretical_stokes_force_v(v).detach()
        x_in = (v - v_min) / (v_max - v_min)
        model = StokesNetV().to(device)
        meta.update({"kind": "v_mms", "v_min": v_min, "v_max": v_max})
    else:
        raise KeyError(key)

    mu = y.mean()
    sigma = y.std().clamp_min(1e-30)
    y_norm = (y - mu) / sigma
    meta["force_mu"] = float(mu.item())
    meta["force_sigma"] = float(sigma.item())

    losses = _train_loop(model, x_in, y_norm, target)
    return TrainBundle(model=model, meta=meta, losses=losses)


def _load_model(key: str, meta: dict, device) -> nn.Module:
    from mechanisms.ARF_PINN_t import ARFNetT
    from mechanisms.ARF_PINN_x import ARFNet
    from mechanisms.STOKES_PINN_t import StokesNetT
    from mechanisms.STOKES_PINN_v import StokesNetV
    from mechanisms.STOKES_PINN_x import StokesNetX

    f_ckpt = float(meta["frequency_Hz"])
    period = 1.0 / f_ckpt

    if key == "ARF_x":
        model = ARFNet(fourier_features=FOURIER_M)
    elif key == "ARF_t":
        model = ARFNetT(period_seconds=period)
    elif key == "Stokes_x":
        model = StokesNetX(fourier_features=FOURIER_M)
    elif key == "Stokes_t":
        model = StokesNetT(period_seconds=period)
    elif key == "Stokes_v":
        model = StokesNetV()
    else:
        raise KeyError(key)

    model.load_state_dict(torch.load(CKPT_DIR / f"{key}.pth", map_location="cpu"))
    return model.to(device).eval()


def _fit_curves(key: str, model: nn.Module, meta: dict, device):
    from mechanisms.ARF_PINN_t import theoretical_time_factor
    from mechanisms.ARF_PINN_x import theoretical_arf
    from mechanisms.STOKES_PINN_t import theoretical_stokes_force_t
    from mechanisms.STOKES_PINN_v import theoretical_stokes_force_v
    from mechanisms.STOKES_PINN_x import theoretical_stokes_force_x

    mu, sigma = meta["force_mu"], meta["force_sigma"]
    kind = meta["kind"]

    with torch.no_grad():
        if kind == "x_mm":
            test = torch.linspace(meta["x_min"], meta["x_max"], 1000, device=device).unsqueeze(1)
            scaled = (test - meta["x_min"]) / (meta["x_max"] - meta["x_min"])
            pred = model(scaled) * sigma + mu
            true = theoretical_arf(test) if key == "ARF_x" else theoretical_stokes_force_x(test)
            axis = (test[:, 0] * 1000.0).cpu().numpy()
            xlab, ylab = "Position x (mm)", "Position factor (normalized)"
        elif kind == "t_us":
            test = torch.linspace(meta["t_min"], meta["t_max"], 1000, device=device).unsqueeze(1)
            scaled = (test - meta["t_min"]) / (meta["t_max"] - meta["t_min"])
            pred = model(scaled) * sigma + mu
            true = theoretical_time_factor(test) if key == "ARF_t" else theoretical_stokes_force_t(test)
            axis = (test[:, 0] * 1e6).cpu().numpy()
            xlab, ylab = "Time (μs)", r"cos($\omega t$) (normalized)"
        else:
            test = torch.linspace(meta["v_min"], meta["v_max"], 1000, device=device).unsqueeze(1)
            scaled = (test - meta["v_min"]) / (meta["v_max"] - meta["v_min"])
            pred = model(scaled) * sigma + mu
            true = theoretical_stokes_force_v(test)
            axis = (test[:, 0] * 1000.0).cpu().numpy()
            xlab, ylab = "Relative velocity (mm/s)", r"$v_x$ (normalized)"

        scale = float(true.abs().max().item()) + 1e-30
        true_n = (true / scale).cpu().numpy().flatten()
        pred_n = (pred / scale).cpu().numpy().flatten()
        if key == "ARF_x":
            true_n = -true_n
            pred_n = -pred_n

    return axis, true_n, pred_n, xlab, ylab


def _cache_valid() -> bool:
    if not CACHE.exists():
        return False
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    if data.get("version") != 3:
        return False
    from mechanisms.ARF import frequency as f_hz

    for key, *_ in SPECS:
        if not (CKPT_DIR / f"{key}.pth").exists():
            return False
        meta_path = CKPT_DIR / f"{key}_meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if abs(float(meta.get("frequency_Hz", -1.0)) - float(f_hz)) > 1.0:
            return False
    return True


def collect_all(retrain_keys: set[str] | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if _cache_valid():
        out = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        out = {"version": 3, "device": str(device), "networks": {}}
        retrain_keys = {key for key, *_ in SPECS}

    if not retrain_keys:
        return out

    print(f"Retraining on {device}: {', '.join(sorted(retrain_keys))}")
    for i, (key, loss_title, fit_title, target) in enumerate(SPECS):
        if key not in retrain_keys:
            continue
        print(f"  [{key}] target={target:.0e}", flush=True)
        torch.manual_seed(100 + i)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(100 + i)
        bundle = _build_and_train(key, target, device)
        torch.save(bundle.model.state_dict(), CKPT_DIR / f"{key}.pth")
        (CKPT_DIR / f"{key}_meta.json").write_text(json.dumps(bundle.meta, indent=2), encoding="utf-8")
        out["networks"][key] = {
            "loss_title": loss_title,
            "fit_title": fit_title,
            "target_loss": target,
            "epochs": len(bundle.losses) - 1,
            "final_loss": bundle.losses[-1],
            "loss_hist": bundle.losses,
            "frequency_Hz": bundle.meta["frequency_Hz"],
        }
        print(
            f"    f={bundle.meta['frequency_Hz']:.0f} Hz  "
            f"epochs={len(bundle.losses)}  final={bundle.losses[-1]:.3e}",
            flush=True,
        )

    out["version"] = 3
    out["device"] = str(device)
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Cached -> {CACHE}")
    return out


def _plot_loss(ax, losses, title, target):
    ax.plot(np.arange(len(losses)), losses, color=THEORY_C, lw=2.0)
    ax.axhline(target, color=PINN_C, ls="--", lw=1.4, label=rf"Target = {target:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.35, which="both")
    ax.legend(fontsize=10, framealpha=0.92)
    ymin = min(min(losses), target) / 5
    ymax = max(max(losses[: min(80, len(losses))]), target * 5, 1e-4)
    ax.set_ylim(ymin, ymax)


def _plot_fit(ax, axis, true_n, pred_n, title, xlab, ylab):
    ax.plot(axis, true_n, color=THEORY_C, lw=2.2, label="Theory")
    ax.plot(axis, pred_n, color=PINN_C, lw=1.8, ls="--", dashes=(4, 2), label="PINN prediction")
    ax.set_xlabel(xlab, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylab, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.92)
    ax.set_ylim(-1.12, 1.12)


def _draw_pair(ax_l, ax_r, key, rec, device):
    meta = json.loads((CKPT_DIR / f"{key}_meta.json").read_text(encoding="utf-8"))
    model = _load_model(key, meta, device)
    axis, true_n, pred_n, xlab, ylab = _fit_curves(key, model, meta, device)
    corr = float(np.corrcoef(true_n, pred_n)[0, 1])
    rmse = float(np.sqrt(np.mean((true_n - pred_n) ** 2)))
    print(f"  fit {key}: corr={corr:.6f}  RMSE={rmse:.4e}")
    _plot_loss(ax_l, rec["loss_hist"], rec["loss_title"], rec["target_loss"])
    _plot_fit(ax_r, axis, true_n, pred_n, rec["fit_title"], xlab, ylab)


def _parse_retrain_keys() -> set[str]:
    if "--retrain-first" in sys.argv:
        return {"ARF_x"}
    if "--retrain" not in sys.argv:
        return set()
    idx = sys.argv.index("--retrain")
    if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
        return {sys.argv[idx + 1]}
    return {key for key, *_ in SPECS}


def main() -> None:
    from mechanisms.ARF import frequency as f_hz

    print(f"Project frequency = {f_hz} Hz")
    keys = _parse_retrain_keys()
    data = collect_all(retrain_keys=keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    keys = [s[0] for s in SPECS]
    for i, key in enumerate(keys, start=1):
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 4.8))
        _draw_pair(ax_l, ax_r, key, data["networks"][key], device)
        fig.tight_layout()
        save_fig(fig, f"fig07_loss_{i:02d}_{key}.png")

    fig, axes = plt.subplots(5, 2, figsize=(14, 22))
    for row, key in enumerate(keys):
        _draw_pair(axes[row, 0], axes[row, 1], key, data["networks"][key], device)
    fig.suptitle(
        "Figure 7. Residual training loss and theory–PINN fit for five FF-PINN sub-networks",
        fontsize=16,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    save_fig(fig, "fig07_training_losses_5panel.png")


if __name__ == "__main__":
    main()
