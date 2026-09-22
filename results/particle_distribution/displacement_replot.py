"""
Replot displacement vs frequency with logarithmic y-axis

This script reads the CSV data and creates an improved plot with:
- Logarithmic y-axis for better visualization of all SPL levels
- Linear regression lines in log space (appears as straight lines on log plot)
- Distinct colors, line styles, and markers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
_RESULTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RESULTS))
from _figures import FIGURES

# Configuration
csv_file = Path(__file__).parent / "displacement_vs_frequency_multi_spl.csv"
output_file = FIGURES / "fig09_displacement_vs_frequency_log_scale.png"

# Define distinct visual styles
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
markers = ['o', 's', '^', 'D', 'v', 'p']


def fit_regression_curve(x, y):
    """Fit a linear regression line in log space (for log-scale plot)"""
    # Take logarithm of y values for fitting in log space
    log_y = np.log(y)
    
    # Linear fit in log space: log(y) = mx + b
    coeffs = np.polyfit(x, log_y, 1)  # Returns [m, b]
    
    # Calculate R-squared in log space
    log_y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((log_y - log_y_pred)**2)
    ss_tot = np.sum((log_y - np.mean(log_y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Return function that converts back to linear space: y = exp(mx + b)
    def regression_func(x_new):
        return np.exp(np.polyval(coeffs, x_new))
    
    return regression_func, r_squared


def main():
    """Main plotting function"""
    print("="*70)
    print("Replotting displacement vs frequency with logarithmic y-axis")
    print("="*70)
    
    # Read CSV data
    print(f"Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Remove 170 dB data
    df = df[df['SPL_dB'] != 170]
    print("Removed 170 dB data")
    
    # Get unique SPL values
    spl_values = sorted(df['SPL_dB'].unique())
    print(f"SPL values: {spl_values}")
    print(f"Total data points: {len(df)}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Plot each SPL
    for idx, spl in enumerate(spl_values):
        # Filter data for this SPL
        spl_data = df[df['SPL_dB'] == spl].sort_values('Frequency_kHz')
        freqs_khz = spl_data['Frequency_kHz'].values
        disps = spl_data['Displacement_mm_per_cycle'].values
        
        # Select style for this SPL
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        
        print(f"\nSPL {spl} dB:")
        print(f"  Frequencies: {freqs_khz}")
        print(f"  Displacements: {disps}")
        print(f"  Style: color={color}, marker={marker}, linestyle={linestyle}")
        
        # Plot data points
        ax.scatter(freqs_khz, disps, s=120, c=color, alpha=0.9,
                  marker=marker, edgecolors='black', linewidth=1.5, zorder=10)
        
        # Fit linear regression
        if len(freqs_khz) >= 2:  # Need at least 2 points for a line
            try:
                reg_func, r_squared = fit_regression_curve(freqs_khz, disps)
                
                # Generate smooth x values for the line
                x_smooth = np.linspace(freqs_khz.min(), freqs_khz.max(), 100)
                y_smooth = reg_func(x_smooth)
                
                # Ensure non-negative
                y_smooth = np.maximum(y_smooth, 1e-10)  # Avoid zero for log scale
                
                # Plot regression line with R² in label (R² computed in log space)
                label = f'SPL = {spl} dB (R² = {r_squared:.3f})'
                print(f"  Linear regression in log space, R² = {r_squared:.3f}")
                
                ax.plot(x_smooth, y_smooth, linestyle=linestyle, color=color,
                       linewidth=3, alpha=0.9, zorder=5, label=label)
            except Exception as e:
                print(f"  Warning: Regression failed: {e}")
                ax.plot(freqs_khz, disps, linestyle=linestyle, color=color,
                       linewidth=3, alpha=0.9, zorder=5, label=f'SPL = {spl} dB')
    
    # Set logarithmic y-axis
    ax.set_yscale('log')
    
    # Format
    ax.set_xlabel('Frequency (kHz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Average Displacement per Cycle (mm/cycle, log scale)', 
                 fontsize=16, fontweight='bold')
    ax.set_title('Particle Average Displacement vs. Frequency',
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
    ax.legend(fontsize=12, loc='best', framealpha=0.95, ncol=1)
    
    # Set x-axis limits with margin
    all_freqs = df['Frequency_kHz'].values
    x_margin = (all_freqs.max() - all_freqs.min()) * 0.08
    ax.set_xlim(all_freqs.min() - x_margin, all_freqs.max() + x_margin)
    
    # Y-axis limits are automatic for log scale, but we can adjust if needed
    all_disps = df['Displacement_mm_per_cycle'].values
    y_min, y_max = all_disps.min(), all_disps.max()
    ax.set_ylim(y_min * 0.5, y_max * 1.5)  # Give some breathing room
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"✓ Figure saved to: {output_file}")
    print(f"{'='*70}")
    
    # Also display
    plt.show()


if __name__ == "__main__":
    main()
