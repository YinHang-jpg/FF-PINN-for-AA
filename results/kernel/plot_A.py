"""Build / refresh ``panel_a_curves.npz`` for Fig. 13 panel (a).

For each frequency in {8, 10, 12, 16} kHz, load the mean orthokinetic
kernel from ``kernel_data_freq_{f}k.npz`` (one acoustic period, averaged
over diameter pairs, scaled by 1e12). If that archive is missing or its
peak is unusable, keep the existing entry in ``panel_a_curves.npz``.

This script only aggregates archived kernel NPZs / the curve cache. It does
not re-run ``kernel_extractor.py`` or any particle simulation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

KERNEL = Path(__file__).resolve().parent
OUT = KERNEL / "panel_a_curves.npz"
FREQS = (8, 10, 12, 16)
# Intact mean-kernel extracts for this panel sit roughly in 10–35 (1e-12 m^3/s).
MIN_PEAK = 1.0
MAX_PEAK = 50.0


def mean_kernel_npz(freq_k: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(KERNEL / f"kernel_data_freq_{freq_k}k.npz")
    t = np.asarray(d["time_points"], dtype=float)
    K = np.asarray(d["kernels"], dtype=float)
    period = 1.0 / (freq_k * 1000.0)
    m = t <= period + 1e-15
    return t[m] * 1e6, K[m].reshape(int(m.sum()), -1).mean(axis=1) * 1e12


def _load_npz(freq_k: int) -> tuple[np.ndarray, np.ndarray]:
    path = KERNEL / f"kernel_data_freq_{freq_k}k.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    t, k = mean_kernel_npz(freq_k)
    peak = float(np.max(k))
    if peak < MIN_PEAK or peak > MAX_PEAK:
        raise ValueError(
            f"{path.name} peak={peak:.3g} outside [{MIN_PEAK}, {MAX_PEAK}]; "
            "re-run results/kernel/kernel_extractor.py for this frequency."
        )
    return t, k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-npz",
        action="store_true",
        help="Require usable kernel_data_freq_*.npz for every frequency "
        "(no cache fallback).",
    )
    args = ap.parse_args()

    cached: dict[str, np.ndarray] = {}
    if OUT.is_file():
        old = np.load(OUT, allow_pickle=True)
        for fk in FREQS:
            key_t, key_k = f"t_{fk}", f"k_{fk}"
            if key_t in old.files and key_k in old.files:
                cached[key_t] = np.asarray(old[key_t], dtype=float)
                cached[key_k] = np.asarray(old[key_k], dtype=float)

    payload: dict[str, np.ndarray] = {}
    for fk in FREQS:
        key_t, key_k = f"t_{fk}", f"k_{fk}"
        used = None
        try:
            t, k = _load_npz(fk)
            payload[key_t], payload[key_k] = t, k
            used = "npz"
        except (FileNotFoundError, ValueError) as exc:
            if args.from_npz:
                raise
            if key_t in cached and key_k in cached:
                payload[key_t], payload[key_k] = cached[key_t], cached[key_k]
                used = "cache"
                print(f"{fk} kHz: NPZ unavailable ({exc}); using cache")
            else:
                raise
        print(
            f"{fk} kHz: source={used}, n={len(payload[key_t])}, "
            f"max={float(payload[key_k].max()):.3f}"
        )

    np.savez(OUT, **payload)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
