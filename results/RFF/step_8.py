"""
Exploratory sweep: epochs to reach MSE = 1e-8 for M = 0, 8, 16, ..., 64.

Does NOT modify fourier_M_composite.png or any manuscript files.
Writes only:
  fourier_M_sweep0_64_step8.json
  fourier_M_sweep0_64_step8.png
  fourier_M_sweep0_64_step8.pdf
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
JSON_PATH = OUT_DIR / "fourier_M_sweep0_64_step8.json"
PNG_PATH = OUT_DIR / "fourier_M_sweep0_64_step8.png"
PDF_PATH = OUT_DIR / "fourier_M_sweep0_64_step8.pdf"

TARGET_LOSS = 1e-8
MAX_EPOCHS = 500_000
PRINT_INTERVAL = 500
M_ORDER = list(range(0, 65, 8))  # 0, 8, 16, 24, 32, 40, 48, 56, 64
HIGHLIGHT_M = 32
SEED_DATA = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M_COLORS = {
    0: "0.55",
    8: "tab:olive",
    16: "tab:green",
    24: "tab:cyan",
    32: "tab:blue",
    40: "tab:orange",
    48: "tab:red",
    56: "tab:pink",
    64: "tab:purple",
}


def build_spatial_data():
    wavelength = sound_speed / frequency
    x_range = 3 * wavelength
    x_min, x_max = -x_range / 2, x_range / 2
    torch.manual_seed(SEED_DATA)
    x = (torch.rand(20000, 1, device=DEVICE) * x_range - x_range / 2).float()
    f = theoretical_stokes_force_x(x).detach()
    xs = (x - x_min) / (x_max - x_min)
    mu, sigma = f.mean(), f.std().clamp_min(1e-30)
    y = (f - mu) / sigma
    return xs, y


def train_one(M: int, xs, y) -> dict:
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
    reached = False

    while epoch <= MAX_EPOCHS:
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
            if epoch % (PRINT_INTERVAL * 10) == 0 or loss_val < TARGET_LOSS:
                lr = optimizer.param_groups[0]["lr"]
                print(f"    Epoch {epoch:06d} | Loss = {loss_val:.3e} | LR = {lr:.2e}", flush=True)

        if loss_val < TARGET_LOSS:
            reached = True
            print(f"    Target {TARGET_LOSS:.1e} at epoch {epoch}", flush=True)
            break
        epoch += 1

    if not reached:
        print(f"    Stopped at cap {MAX_EPOCHS} epochs, final loss = {loss_val:.3e}", flush=True)

    return {
        "M": M,
        "loss_hist": hist,
        "epochs": int(epoch if reached else MAX_EPOCHS),
        "final_loss": loss_val,
        "reached_target": reached,
        "train_s": round(time.perf_counter() - t0, 2),
        "input_dim": int(M * 2 + 1),
    }


def load_results() -> list[dict]:
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        if float(data.get("target", 0)) == TARGET_LOSS:
            return data.get("models", [])
    return []


def save_results(models: list[dict]) -> None:
    JSON_PATH.write_text(
        json.dumps(
            {
                "seed_data": SEED_DATA,
                "target": TARGET_LOSS,
                "max_epochs": MAX_EPOCHS,
                "M_order": M_ORDER,
                "branch": "StokesNetX / cos(kx); M=0 means no RFF (input dim = 1)",
                "device": str(DEVICE),
                "models": models,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def render_bar(models: list[dict]) -> None:
    by_M = {m["M"]: m for m in models}
    Ms = [M for M in M_ORDER if M in by_M]
    epochs = np.array([by_M[M]["epochs"] if by_M[M]["reached_target"] else np.nan for M in Ms], dtype=float)
    reached = [by_M[M]["reached_target"] for M in Ms]

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": 13,
            "axes.labelsize": 15,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    valid = np.isfinite(epochs)
    y_base = float(np.nanmin(epochs[valid]) * 0.15) if valid.any() else 1.0

    for M, ep, ok in zip(Ms, epochs, reached):
        if not ok:
            ax.bar(
                M,
                y_base * 3,
                bottom=y_base,
                width=6.0,
                color="0.85",
                edgecolor="black",
                hatch="//",
                linewidth=1.0,
                label="_nolegend_",
            )
            ax.text(M, y_base * 4, "cap", ha="center", va="bottom", fontsize=9, color="0.35")
            continue
        h = ep - y_base
        is_hi = M == HIGHLIGHT_M
        ax.bar(
            M,
            h,
            bottom=y_base,
            width=6.0,
            color=M_COLORS.get(M, "0.5"),
            edgecolor="black",
            linewidth=1.6 if is_hi else 1.0,
            alpha=1.0 if is_hi else 0.92,
            label=rf"$M$ = {M}" + (" (current)" if is_hi else ""),
        )

    ax.axvline(HIGHLIGHT_M, color="tab:blue", ls="--", lw=1.2, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlabel(r"Fourier feature dimension $M$", fontweight="bold")
    ax.set_ylabel(r"Epochs to reach MSE $=10^{-8}$", fontweight="bold")
    ax.set_xticks(Ms)
    ax.set_xlim(-4, 68)
    if valid.any():
        ymax = float(np.nanmax(epochs) * 2.5)
        ax.set_ylim(y_base, ymax)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2)
    ax.set_title("Exploratory sweep: M = 0, 8, 16, ..., 64 (step 8)", fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white", dpi=600)
    fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [figure] {PNG_PATH.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    models = load_results() if args.resume else []
    done = {m["M"] for m in models}

    print(f"device={DEVICE}  target={TARGET_LOSS:.0e}  M={M_ORDER}", flush=True)
    xs, y = build_spatial_data()

    for M in M_ORDER:
        if M in done:
            print(f"skip M={M}", flush=True)
            continue
        print(f"Training M={M} (input dim = {M*2+1}) ...", flush=True)
        res = train_one(M, xs, y)
        print(
            f"  done M={M}: epochs={res['epochs']} final={res['final_loss']:.3e} "
            f"reached={res['reached_target']} time={res['train_s']}s",
            flush=True,
        )
        models.append(res)
        models.sort(key=lambda r: M_ORDER.index(r["M"]))
        save_results(models)
        render_bar(models)

    print("\nSummary (M, epochs, reached):", flush=True)
    for m in models:
        tag = "OK" if m["reached_target"] else "CAP"
        print(f"  M={m['M']:2d}  epochs={m['epochs']:7d}  {tag}  final={m['final_loss']:.3e}", flush=True)

    if models:
        ok = [m for m in models if m["reached_target"]]
        if ok:
            best = min(ok, key=lambda m: m["epochs"])
            print(f"\nFastest among finished runs: M={best['M']} at {best['epochs']} epochs", flush=True)


if __name__ == "__main__":
    main()
