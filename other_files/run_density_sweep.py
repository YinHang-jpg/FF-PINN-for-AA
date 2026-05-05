#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin launcher for the density sweep driver.
"""
import sys
from pathlib import Path

# Repo root on sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Run density_distribution_plot
if __name__ == "__main__":
    # chdir to repo root
    import os
    os.chdir(str(project_root))
    
    from results.density_sweep import density_distribution_plot
    density_distribution_plot.main()
