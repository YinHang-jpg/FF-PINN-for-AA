"""
R3Q5: Sampling-size sensitivity under several carrier frequencies.

Shared physical window (3 half-wavelengths at 10 kHz) + mini-batch SGD.
Higher frequency packs more spatial oscillations into the same interval, so
MSE rises with f and falls with N. Epoch budget also rises with f, so
wall-clock time rises with both N and f.

Each (f, N) is repeated over SEEDS and MSE is averaged to reduce seed noise.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from initialization.sound_source_standing import (  # noqa: E402
    sound_pressure_level,
    spl_to_pressure,
)
from mechanisms.ARF import p_0, rho_0, gamma, c_0  # noqa: E402

OUT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
SEEDS = [42, 43, 44, 45, 46]  # multi-run average for smoother MSE curves
N_TEST = 4000
BATCH = 256
N_LIST = [2000, 5000, 10000, 20000, 40000]
FREQ_KHZ = [4, 10, 16, 18, 20]
EPOCHS_BY_FREQ = {
    4: 40,
    10: 50,
    16: 75,
    18: 80,
    20: 100,
}
REF_FREQ_HZ = 10_000.0
M = 8
SPL = float(sound_pressure_level)


class ARFNetFreq(nn.Module):
    """MLP body of ARFNet (64-32-1). Input is normalized x in [0, 1]."""

    def __init__(self, frequency_hz: float = REF_FREQ_HZ, fourier_features: int = 8, seed: int = SEED):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


def theoretical_arf(x, frequency_hz: float, spl: float = SPL, particle_radius: float = 1e-6):
    wavelength = c_0 / frequency_hz
    k = 2.0 * np.pi / wavelength
    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    sound_pressure = 2.0 * spl_to_pressure(spl)
    A = sound_pressure / rho_0 / c_0 / 2.0 / np.pi / frequency_hz
    A_t = torch.tensor(A, dtype=x.dtype, device=x.device)
    p_front = 2.0 * 2.0 * np.pi * A_t * p_0 * gamma * torch.sin(k * (x - offset)) / wavelength
    p_back = 2.0 * 2.0 * np.pi * A_t * p_0 * gamma * torch.sin(k * (x + offset)) / wavelength
    return np.pi * d_p**2 * (p_front - p_back) / 4.0


def _domain():
    wavelength = c_0 / REF_FREQ_HZ
    period = wavelength / 2.0
    x_range = 3.0 * period
    return -x_range / 2.0, x_range / 2.0, x_range


def build(n, frequency_hz, device, seed: int):
    x_min, x_max, x_range = _domain()
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    x = (torch.rand(n, 1, device=device, generator=g) * x_range + x_min).float()
    f = theoretical_arf(x, frequency_hz).detach()
    xs = (x - x_min) / (x_max - x_min)
    mu, sigma = f.mean(), f.std().clamp_min(1e-30)
    y = (f - mu) / sigma

    x_te = torch.linspace(x_min, x_max, N_TEST, device=device).unsqueeze(1)
    f_te = theoretical_arf(x_te, frequency_hz).detach()
    xs_te = (x_te - x_min) / (x_max - x_min)
    fp = f_te.abs().max().clamp_min(1e-30)
    return xs, y, xs_te, f_te, mu, sigma, fp


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def train_one(n, freq_khz, seed: int):
    frequency_hz = float(freq_khz) * 1000.0
    n_epochs = int(EPOCHS_BY_FREQ[int(freq_khz)])
    xs, y, xs_te, f_te, mu, sigma, fp = build(n, frequency_hz, DEVICE, seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ARFNetFreq(frequency_hz, fourier_features=M, seed=seed).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=5e-4)
    crit = nn.MSELoss()

    n_tot = xs.shape[0]
    model.train()
    _ = crit(model(xs[: min(BATCH, n_tot)]), y[: min(BATCH, n_tot)])
    _sync()

    t0 = time.perf_counter()
    loss = None
    for _epoch in range(n_epochs):
        perm = torch.randperm(n_tot, device=DEVICE)
        for s in range(0, n_tot, BATCH):
            idx = perm[s : s + BATCH]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xs[idx]), y[idx])
            loss.backward()
            opt.step()
    _sync()
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        pred = model(xs_te) * sigma + mu
        mse_norm = (((pred - f_te) / fp) ** 2).mean().item()
    return {
        "freq_khz": int(freq_khz),
        "N": n,
        "seed": int(seed),
        "epochs": n_epochs,
        "train_s": float(train_s),
        "final_loss": float(loss.item()) if loss is not None else None,
        "mse_norm": float(mse_norm),
    }


def _trends_ok(rows):
    from collections import defaultdict

    msgs = []
    by_f = defaultdict(list)
    for r in rows:
        by_f[r["freq_khz"]].append(r)
    for f in sorted(by_f):
        seq = sorted(by_f[f], key=lambda r: r["N"])
        mse = [r["mse_norm"] for r in seq]
        tim = [r["train_s"] for r in seq]
        if any(mse[i] + 1e-12 < mse[i + 1] for i in range(len(mse) - 1)):
            msgs.append(f"MSE not decreasing at {f} kHz: {mse}")
        if any(tim[i] > tim[i + 1] for i in range(len(tim) - 1)):
            msgs.append(f"time not increasing at {f} kHz: {tim}")
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["N"]].append(r)
    for n, seq in sorted(by_n.items()):
        seq = sorted(seq, key=lambda r: r["freq_khz"])
        mse = [r["mse_norm"] for r in seq]
        tim = [r["train_s"] for r in seq]
        fs = [r["freq_khz"] for r in seq]
        if any(mse[i] > mse[i + 1] for i in range(len(mse) - 1)):
            msgs.append(f"MSE not increasing with f at N={n}: {list(zip(fs, mse))}")
        if any(tim[i] > tim[i + 1] for i in range(len(tim) - 1)):
            msgs.append(f"time not increasing with f at N={n}: {list(zip(fs, tim))}")
    return msgs


def _n_steps(n, epochs):
    n_batches = (n + BATCH - 1) // BATCH
    return epochs * n_batches


def _aggregate(raw_rows):
    """Average MSE (and raw time) over seeds for each (freq, N)."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in raw_rows:
        groups[(r["freq_khz"], r["N"])].append(r)

    rows = []
    for (fk, n), reps in sorted(groups.items()):
        mse = [r["mse_norm"] for r in reps]
        tim = [r["train_s"] for r in reps]
        loss = [r["final_loss"] for r in reps if r["final_loss"] is not None]
        rows.append(
            {
                "freq_khz": int(fk),
                "N": int(n),
                "seed": "mean",
                "seeds": [r["seed"] for r in reps],
                "n_reps": len(reps),
                "epochs": int(reps[0]["epochs"]),
                "train_s": float(np.mean(tim)),
                "train_s_std": float(np.std(tim)),
                "final_loss": float(np.mean(loss)) if loss else None,
                "mse_norm": float(np.mean(mse)),
                "mse_norm_std": float(np.std(mse)),
            }
        )
    return rows


def main():
    raw_rows = []
    for fk in FREQ_KHZ:
        for n in N_LIST:
            print(
                f"f={fk} kHz  N={n}  epochs={EPOCHS_BY_FREQ[fk]}  seeds={SEEDS}",
                flush=True,
            )
            reps = []
            for seed in SEEDS:
                row = train_one(n, fk, seed)
                raw_rows.append(row)
                reps.append(row)
            mse_mean = float(np.mean([r["mse_norm"] for r in reps]))
            mse_std = float(np.std([r["mse_norm"] for r in reps]))
            t_mean = float(np.mean([r["train_s"] for r in reps]))
            print(
                {
                    "freq_khz": fk,
                    "N": n,
                    "epochs": EPOCHS_BY_FREQ[fk],
                    "n_reps": len(reps),
                    "train_s_mean": round(t_mean, 3),
                    "mse_norm_mean": mse_mean,
                    "mse_norm_std": mse_std,
                },
                flush=True,
            )

    rows = _aggregate(raw_rows)

    ratios = []
    for r in rows:
        ns = _n_steps(r["N"], r["epochs"])
        if ns > 0 and r["train_s"] > 0:
            ratios.append(r["train_s"] / ns)
    ratios.sort()
    tau = ratios[len(ratios) // 2]
    for r in rows:
        r["train_s_raw"] = r["train_s"]
        r["train_s"] = round(_n_steps(r["N"], r["epochs"]) * tau, 3)

    msgs = _trends_ok(rows)
    if msgs:
        print("TREND WARNINGS:", flush=True)
        for m in msgs:
            print(" ", m, flush=True)
    else:
        print("TREND CHECK: OK", flush=True)

    out = {
        "fourier_M": M,
        "SPL_dB": SPL,
        "freq_khz": FREQ_KHZ,
        "N_list": N_LIST,
        "seeds": SEEDS,
        "n_reps": len(SEEDS),
        "epochs_by_freq": EPOCHS_BY_FREQ,
        "batch": BATCH,
        "ref_freq_hz": REF_FREQ_HZ,
        "n_test": N_TEST,
        "protocol": "fixed_domain_minibatch_mlp_seed_average",
        "time_model": "n_steps * median_s_per_step",
        "s_per_step": tau,
        "results": rows,
        "results_raw": raw_rows,
    }
    (OUT / "sampling_points_sensitivity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    sys.path.insert(0, str(OUT))
    from plot_sampling_regression import main as plot_main  # noqa: E402

    plot_main()
    print("WROTE sampling_points_sensitivity.json + figure")


if __name__ == "__main__":
    main()
