"""Shared paths for Fig.7 scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root (HERE = results/Fig7)
DATA = HERE / "data"
FIGURES = ROOT / "figures"
DPI = 400

FIGURES.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAINING_TARGETS = {
    "ARF_x": 1e-6,
    "ARF_t": 1e-7,
    "Stokes_x": 1e-8,
    "Stokes_t": 1e-8,
    "Stokes_v": 1e-8,
}


def save_fig(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    path = FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")
    return path
