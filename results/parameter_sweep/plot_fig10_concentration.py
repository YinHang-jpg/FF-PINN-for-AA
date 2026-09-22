"""
Figure 10 — concentration vs frequency after 100 / 1000 acoustic cycles.

Publishes the authenticated outputs of ``freq_sweep._draw_concentration_combined_plot``
(``concentration_distribution_combined_{100,1000}cycles.png``) into ``figures/``
under the manuscript filenames. Those PNGs are the same combined plots used in the
manuscript (44000 particles, SPL=168.5 dB, frequencies 8/10/12/16 kHz).

To regenerate the source PNGs from simulation (≈20 min):
    python results/parameter_sweep/freq_sweep.py

Outputs (under figures/):
  fig10_concentration_100cycles.png
  fig10_concentration_1000cycles.png
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "results"))
from _figures import FIGURES  # noqa: E402

SOURCES = {
    100: HERE / "_manuscript_fig10_source" / "concentration_distribution_combined_100cycles.png",
    1000: HERE / "_manuscript_fig10_source" / "concentration_distribution_combined_1000cycles.png",
}
# Fallback to working copies next to this script
_FALLBACK = {
    100: HERE / "concentration_distribution_combined_100cycles.png",
    1000: HERE / "concentration_distribution_combined_1000cycles.png",
}
TARGETS = {
    100: "fig10_concentration_100cycles.png",
    1000: "fig10_concentration_1000cycles.png",
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for cycles, src in SOURCES.items():
        if not src.is_file():
            src = _FALLBACK[cycles]
        if not src.is_file():
            raise FileNotFoundError(
                f"Missing Fig. 10 source for {cycles} cycles. "
                "Expected under results/parameter_sweep/_manuscript_fig10_source/ "
                "or concentration_distribution_combined_{cycles}cycles.png. "
                "To regenerate from simulation (needs complete PINN/freq_*k packs "
                "including normalization JSON): python results/parameter_sweep/freq_sweep.py"
            )
        dst = FIGURES / TARGETS[cycles]
        shutil.copy2(src, dst)
        print("WROTE", dst, f"({dst.stat().st_size} bytes from {src})")


if __name__ == "__main__":
    main()
