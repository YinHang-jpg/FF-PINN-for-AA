#!/usr/bin/env python
"""Thin launcher for the density-sweep driver (Fig. 11)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

if __name__ == "__main__":
    from results.density_sweep import density_distribution_plot

    density_distribution_plot.main()
