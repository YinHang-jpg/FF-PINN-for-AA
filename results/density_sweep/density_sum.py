import sys
from pathlib import Path
_RESULTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RESULTS))
from _figures import FIGURES
"""
Plot particle concentration curves for different particle densities (1000-5000 kg/m³)
Only plot curves in the range x = 2mm to x = 32mm
Y-axis is concentration (%)
All text in English
Plotting style matches comsol_dualplot.py for consistency
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Set matplotlib parameters to match comsol_dualplot.py
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

# Directory containing the data files
data_dir = Path(__file__).parent

# Find all density curve files
density_files = sorted(data_dir.glob('density_curve_density_*.txt'))

# Extract density values and create a mapping
density_data = {}
for file in density_files:
    # Extract density value from filename (e.g., density_curve_density_1000.txt -> 1000)
    match = re.search(r'density_(\d+)\.txt$', file.name)
    if match:
        density = int(match.group(1))
        density_data[density] = file

# Sort by density value
sorted_densities = sorted(density_data.keys())

# Color scheme for different densities (using a gradient)
colors = {
    1000: '#1f77b4',  # blue
    2000: '#ff7f0e',  # orange
    3000: '#2ca02c',  # green
    4000: '#d62728',  # red
    5000: '#9467bd',  # purple
}

# Line styles for different densities - matching comsol_dualplot.py
line_styles = {
    1000: '-',      # solid
    2000: '--',     # dashed
    3000: '-.',     # dash-dot
    4000: ':',      # dotted
    5000: '-',      # solid
}

# Create figure - use same size as comsol_dualplot.py
fig, ax = plt.subplots(figsize=(12, 7))

# Plot each density
for density in sorted_densities:
    data_file = density_data[density]
    
    print(f"Processing: {data_file.name} (Density: {density} kg/m³)")
    
    # Load data: column 0 is x (mm), column 1 is concentration value
    data = np.loadtxt(data_file)
    x = data[:, 0]
    concentration = data[:, 1]
    
    # Filter data: only keep x in range [2, 32] mm
    mask = (x >= 2.0) & (x <= 32.0)
    x_filtered = x[mask]
    concentration_filtered = concentration[mask]
    
    # Get color and line style
    color = colors.get(density, None)
    linestyle = line_styles.get(density, '-')
    
    # Plot with style matching comsol_dualplot.py
    # No additional smoothing - use data directly for accurate representation
    ax.plot(x_filtered, concentration_filtered, 
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            alpha=0.9,
            label=f'{density} kg/m³')
    
    print(f"  Range: [{np.min(concentration_filtered):.2f}%, {np.max(concentration_filtered):.2f}%]")

# Set labels and title - style matching comsol_dualplot.py
ax.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Relative Density Change (%)', fontsize=14, fontweight='bold')
ax.set_title('Particle Density Distribution at Different Particle Densities', 
             fontsize=16, fontweight='bold')

# Set x-axis limits
ax.set_xlim(2, 32)

# Add grid - style matching comsol_dualplot.py
ax.grid(True, alpha=0.3, linestyle='--')

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

# Add legend - style matching comsol_dualplot.py
ax.legend(loc='best', fontsize=12, framealpha=0.9)

# Tight layout
plt.tight_layout()

# Save figure
output_file = FIGURES / "fig11_density_sweep_summary.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_file}")

plt.close()
print("Done!")

