"""
Standalone panel (a): Stokes spatial-factor PINN (STOKES_PINN_x.py), target MSE = 1e-8.

Each M in {8, 16, 32, 48, 64} is trained independently and serially; the figure
is re-rendered after every finished run so partial progress is always visible.

Writes:
  fourier_M_panelA_target1e-8.png / .pdf
  fourier_M_panelA_target1e-8.json   (loss histories, appended per M)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mechanisms.STOKES_PINN_x import (  # noqa: E402
    StokesNetX,
    theoretical_stokes_force_x,
    sound_speed,
    frequency,
)

OUT_DIR = Path(__file__).resolve().parent
RESULT_PATH = OUT_DIR / "fourier_M_panelA_target1e-8.json"
PNG_PATH = OUT_DIR / "fourier_M_panelA_target1e-8.png"
PDF_PATH = OUT_DIR / "fourier_M_panelA_target1e-8.pdf"

TARGET_LOSS = 1e-8
# Match mechanisms/STOKES_PINN_x.py: no epoch cap; train until target.
PRINT_INTERVAL = 500
M_ORDER = [8, 16, 32, 48, 64]
RECOMMENDED_M = 8
M_COLORS = {
    8: "tab:olive",
    16: "tab:green",
    32: "tab:blue",
    48: "tab:red",
    64: "tab:purple",
}
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_spatial_data():
    """Same sampling as mechanisms/STOKES_PINN_x.py main()."""
    wavelength = sound_speed / frequency
    x_range = 3 * wavelength
    x_min, x_max = -x_range / 2, x_range / 2
    torch.manual_seed(SEED)
    x = (torch.rand(20000, 1, device=DEVICE) * x_range - x_range / 2).float()
    f = theoretical_stokes_force_x(x).detach()
    xs = (x - x_min) / (x_max - x_min)
    mu, sigma = f.mean(), f.std().clamp_min(1e-30)
    y = (f - mu) / sigma
    return xs, y


def train_one(M: int, xs, y) -> dict:
    """Same loop as mechanisms/STOKES_PINN_x.py (Adam 5e-4, plateau patience=5000, while True).

    Do not re-seed before model construction: original STOKES_PINN_x.py does not
    call torch.manual_seed, so Fourier weights are a fresh draw each run.
    """
    model = StokesNetX(fourier_features=M).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5000
    )
    criterion = nn.MSELoss()

    hist: list[list[float]] = []
    t0 = time.perf_counter()
    epoch = 0
    loss_val = float("nan")

    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(xs)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        scheduler.step(loss_val)

        if epoch % PRINT_INTERVAL == 0 or loss_val < TARGET_LOSS:
            hist.append([float(epoch), loss_val])
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:05d} | Loss = {loss_val:.3e} | LR = {lr:.2e}", flush=True)
        if loss_val < TARGET_LOSS:
            print(f"    Target loss {TARGET_LOSS:.1e} at epoch {epoch}", flush=True)
            break
        epoch += 1

    return {
        "M": M,
        "loss_hist": hist,
        "epochs": int(epoch),
        "final_loss": loss_val,
        "reached_target": True,
        "train_s": round(time.perf_counter() - t0, 2),
    }


def load_results() -> list[dict]:
    if RESULT_PATH.exists():
        data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if float(data.get("target", 0)) == TARGET_LOSS:
            return data.get("models", [])
    return []


def save_results(models: list[dict]) -> None:
    RESULT_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "target": TARGET_LOSS,
                "scheduler": "ReduceLROnPlateau patience=5000 (STOKES_PINN_x.py)",
                "branch": "StokesNetX / cos(kx)",
                "device": str(DEVICE),
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def render_panel_a(models: list[dict]) -> None:
    """Redraw the standalone panel (a) from whatever runs are finished so far."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.0,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    by_M = {m["M"]: m for m in models}

    for M in M_ORDER:
        if M not in by_M:
            continue
        h = np.asarray(by_M[M]["loss_hist"], dtype=float)
        lw = 2.6 if M == RECOMMENDED_M else 1.8
        ax.semilogy(h[:, 0], h[:, 1], color=M_COLORS[M], lw=lw, label=rf"$M$ = {M}")

    ax.axhline(TARGET_LOSS, color="0.40", ls="--", lw=1.3, label=r"Target $10^{-8}$")
    ax.set_xlabel("Training epoch", fontweight="bold")
    ax.set_ylabel("MSE loss", fontweight="bold")
    ax.set_ylim(TARGET_LOSS / 3.0, 2.0)
    ax.set_xlim(left=0)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.tick_params(width=1.0, length=5)

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white", dpi=600)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    done = ", ".join(f"M={m['M']}" for m in models)
    print(f"  [figure updated] {PNG_PATH.name}  ({done})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="keep already-finished M runs instead of starting fresh")
    args = parser.parse_args()

    models = load_results() if args.resume else []
    done_M = {m["M"] for m in models}

    print(f"device={DEVICE}  target={TARGET_LOSS:.0e}  branch=StokesNetX  patience=5000", flush=True)
    xs, y = build_spatial_data()

    for M in M_ORDER:
        if M in done_M:
            print(f"skip M={M} (already in cache)", flush=True)
            continue
        print(f"Training M={M} ...", flush=True)
        res = train_one(M, xs, y)
        status = "reached" if res["reached_target"] else "capped"
        print(
            f"  M={M}: epochs={res['epochs']} final={res['final_loss']:.3e} "
            f"({status}) {res['train_s']}s",
            flush=True,
        )
        models.append(res)
        models.sort(key=lambda r: M_ORDER.index(r["M"]))
        save_results(models)
        render_panel_a(models)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
