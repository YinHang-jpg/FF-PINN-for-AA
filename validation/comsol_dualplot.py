# Fix Windows Chinese encoding issue
import sys
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

"""
COMSOL Particle Position Data Visualization Tool

Features:
    1. Read CSV format particle position data
    2. Calculate relative particle density change (percentage) using Gaussian kernel density estimation
    3. Generate two plots showing relative density change compared to initial state:
       - Plot 1: t=0.01, 0.03, 0.05s
       - Plot 2: t=0.06, 0.08, 0.1s
    
    *** Plotting logic optimized for comparable peak values with PD_time.py ***
    - Uses sigma=0.3 for Gaussian kernel (balanced between 0.2 and 0.5)
    - NO additional smoothing (gaussian_filter1d removed)
    - alpha=0.85 for line transparency (consistent with PD_time.py)

CSV Data Format Requirements:
    - First column: Time (unit: seconds)
    - Subsequent columns: Position of each particle on x-axis (unit: millimeters, range 0-34mm)
    - Rows: Arbitrary (each row corresponds to a time point)
    - Columns: Arbitrary (first column is time, subsequent columns are particle positions)

Example Data Structure:
    time,particle_1,particle_2,particle_3,...
    0.000,5.2,8.1,15.3,...
    0.001,5.3,8.2,15.1,...
    0.002,5.5,8.3,14.9,...
    ...

Usage:
    python validation/comsol_dualplot.py

Output:
    - validation/comsol_dualplot_1.png  (relative density change at t=0.01, 0.03, 0.05s, x=2-32mm)
    - validation/comsol_dualplot_2.png  (relative density change at t=0.06, 0.08, 0.1s, x=2-32mm)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_particle_density(particle_positions_mm, x_grid_mm, sigma=0.3):
    """
    Calculate particle number density using Gaussian kernel density estimation
    (Balanced smoothing: sigma=0.3, between PD_time.py's 0.2 and original 0.5)
    
    Parameters:
        particle_positions_mm: 1D array containing x-coordinates of all particles (unit: mm)
        x_grid_mm: 1D array of x-axis grid points for density calculation (unit: mm)
        sigma: Standard deviation of Gaussian kernel (unit: mm), controls smoothness
               Default: 0.3 (balanced between 0.2 and 0.5 for comparable peak values)
    
    Returns:
        density: Particle number density at each point in x_grid_mm
    
    Notes:
        - Uses Gaussian kernel: exp(-distance²/(2*sigma²))
        - Automatically filters NaN and out-of-range values
        - Works with arbitrary number of particles
    """
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    # Remove invalid values (NaN or out-of-range values)
    particle_positions_mm = particle_positions_mm[~np.isnan(particle_positions_mm)]
    particle_positions_mm = particle_positions_mm[(particle_positions_mm >= 0) & (particle_positions_mm <= 34)]
    
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    density = np.zeros_like(x_grid_mm)
    
    # Gaussian kernel density estimation
    for idx, x_pos in enumerate(x_grid_mm):
        # Calculate distance from all particles to current position
        distances = np.abs(particle_positions_mm - x_pos)
        
        # Calculate density contribution using Gaussian kernel
        density_contribution = np.exp(-distances**2 / (2 * sigma**2))
        density[idx] = np.sum(density_contribution)
    
    return density


def main():
    """
    主函数：读取COMSOL数据并绘制粒子数密度随时间的变化
    
    功能：
    - 自动检测CSV文件中的时间点数量（第一列）
    - 自动处理任意数量的粒子（后续列）
    - 生成两张图：第一张包含t=0.01, 0.03, 0.05s，第二张包含t=0.06, 0.08, 0.1s
    """
    
    print("="*80)
    print("COMSOL Particle Position Data Visualization")
    print("="*80)
    
    # Read CSV file
    csv_path = 'validation/comsol_positions.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return
    
    print(f"\nReading file: {csv_path}")
    # Use header=None to read all rows as data (not treating first row as column names)
    df = pd.read_csv(csv_path, header=None)
    
    # Auto-detect number of timesteps and particles
    num_timesteps = len(df)
    num_particles = len(df.columns) - 1
    
    print(f"Data dimensions: {df.shape}")
    print(f"✓ Detected {num_timesteps} timesteps")
    print(f"✓ Detected {num_particles} particles")
    
    # Validate data
    if num_timesteps == 0:
        print("Error: No data rows in CSV file")
        return
    if num_particles == 0:
        print("Error: No particle position columns in CSV file (excluding time column)")
        return
    
    # First column is time
    time_values = df.iloc[:, 0].values
    
    # Check for NaN values in time column
    nan_count = np.sum(np.isnan(time_values))
    if nan_count > 0:
        print(f"\n⚠️  Warning: Detected {nan_count} NaN values in time column!")
        print(f"   Valid timesteps: {num_timesteps - nan_count}")
        print(f"   Invalid timestep positions: {np.where(np.isnan(time_values))[0].tolist()}")
        
        # Filter out NaN timesteps
        valid_indices = ~np.isnan(time_values)
        time_values_valid = time_values[valid_indices]
        particle_positions_valid = df.iloc[:, 1:].values[valid_indices, :]
        
        print(f"\n   Automatically filtered invalid timesteps, processing {len(time_values_valid)} valid timesteps")
        
        # Update data
        time_values = time_values_valid
        particle_positions = particle_positions_valid
        num_timesteps = len(time_values)
        
        if num_timesteps == 0:
            print("\nError: No valid timesteps after filtering!")
            return
    else:
        # No NaN values, read particle positions normally
        particle_positions = df.iloc[:, 1:].values  # shape: (n_timesteps, n_particles)
    
    print(f"\nTime range: {time_values[0]:.6f}s to {time_values[-1]:.6f}s")
    
    # Print timesteps based on count
    if num_timesteps <= 20:
        print(f"Timesteps: {time_values}")
    else:
        print(f"Timesteps (first 5): {time_values[:5]}")
        print(f"Timesteps (last 5): {time_values[-5:]}")
    
    # Define x-axis grid (for density calculation)
    x_grid_mm = np.linspace(0, 34, 200)  # 0 to 34mm, 200 points
    
    # Set plotting parameters
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Define target times for two plots
    target_times_plot1 = [0.01, 0.03, 0.05]  # First plot times
    target_times_plot2 = [0.06, 0.08, 0.1]   # Second plot times
    
    # Define line styles and colors for each curve
    line_styles = ['-', '--', '-.']  # solid, dashed, dash-dot
    colors_plot1 = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    colors_plot2 = ['#d62728', '#9467bd', '#8c564b']  # red, purple, brown
    
    print(f"\n{'='*80}")
    print(f"Calculating relative particle density change...")
    print(f"{'='*80}\n")
    
    # Function to find closest time index
    def find_closest_time_index(target_time, time_array):
        """Find the index of the closest time value in the array"""
        return np.argmin(np.abs(time_array - target_time))
    
    # Calculate initial density (at t=0) - using sigma=0.3 for balanced smoothing
    initial_positions = particle_positions[0, :]
    initial_density = calculate_particle_density(initial_positions, x_grid_mm, sigma=0.3)
    
    print(f"Initial density calculated at t={time_values[0]:.6f}s")
    print(f"Valid particles at t=0: {np.sum(~np.isnan(initial_positions))}")
    
    # Create mask for x-axis range (2-32mm)
    x_mask = (x_grid_mm >= 2) & (x_grid_mm <= 32)
    x_grid_cropped = x_grid_mm[x_mask]
    
    # ==================== Generate Plot 1: t=0.01, 0.03, 0.05s ====================
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    print(f"\nGenerating Plot 1 (t = 0.01, 0.03, 0.05s)...")
    for idx, target_t in enumerate(target_times_plot1):
        # Find closest time index
        time_idx = find_closest_time_index(target_t, time_values)
        actual_t = time_values[time_idx]
        
        # Get particle positions at this time
        positions_at_t = particle_positions[time_idx, :]
        
        # Calculate particle density (using sigma=0.3 for balanced smoothing)
        density = calculate_particle_density(positions_at_t, x_grid_mm, sigma=0.3)
        
        # Calculate relative change (percentage) - no smoothing, consistent with PD_time.py
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (density - initial_density) / initial_density * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Crop to 2-32mm range
        relative_change_cropped = relative_change[x_mask]
        
        # Plot relative change curve (alpha=0.85, consistent with PD_time.py)
        label = f't = {actual_t:.2f}s'
        ax1.plot(x_grid_cropped, relative_change_cropped, 
                color=colors_plot1[idx], 
                linestyle=line_styles[idx],
                linewidth=2.5, 
                alpha=0.9,
                label=label)
        
        print(f"  Time {idx+1}/3: t={actual_t:.4f}s (target: {target_t}s), "
              f"valid particles={np.sum(~np.isnan(positions_at_t))}, "
              f"relative change range=[{np.min(relative_change_cropped):.2f}%, {np.max(relative_change_cropped):.2f}%]")
    
    # Set plot properties
    ax1.set_xlim(2, 32)
    ax1.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Relative Density Change (%)', fontsize=14, fontweight='bold')
    ax1.set_title('COMSOL Data: Relative Particle Density Change', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', fontsize=12, framealpha=0.9)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Save Plot 1
    output_dir = 'validation'
    os.makedirs(output_dir, exist_ok=True)
    output_path1 = os.path.join(output_dir, 'comsol_dualplot_1.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    
    print(f"\nPlot 1 saved to: {output_path1}")
    plt.close()
    
    # ==================== Generate Plot 2: t=0.06, 0.08, 0.1s ====================
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    print(f"\nGenerating Plot 2 (t = 0.06, 0.08, 0.1s)...")
    for idx, target_t in enumerate(target_times_plot2):
        # Find closest time index
        time_idx = find_closest_time_index(target_t, time_values)
        actual_t = time_values[time_idx]
        
        # Get particle positions at this time
        positions_at_t = particle_positions[time_idx, :]
        
        # Calculate particle density (using sigma=0.3 for balanced smoothing)
        density = calculate_particle_density(positions_at_t, x_grid_mm, sigma=0.3)
        
        # Calculate relative change (percentage) - no smoothing, consistent with PD_time.py
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (density - initial_density) / initial_density * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Crop to 2-32mm range
        relative_change_cropped = relative_change[x_mask]
        
        # Plot relative change curve (alpha=0.85, consistent with PD_time.py)
        label = f't = {actual_t:.2f}s'
        ax2.plot(x_grid_cropped, relative_change_cropped, 
                color=colors_plot2[idx], 
                linestyle=line_styles[idx],
                linewidth=2.5, 
                alpha=0.9,
                label=label)
        
        print(f"  Time {idx+1}/3: t={actual_t:.4f}s (target: {target_t}s), "
              f"valid particles={np.sum(~np.isnan(positions_at_t))}, "
              f"relative change range=[{np.min(relative_change_cropped):.2f}%, {np.max(relative_change_cropped):.2f}%]")
    
    # Set plot properties
    ax2.set_xlim(2, 32)
    ax2.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Relative Density Change (%)', fontsize=14, fontweight='bold')
    ax2.set_title('COMSOL Data: Relative Particle Density Change', 
                 fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', fontsize=12, framealpha=0.9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Save Plot 2
    output_path2 = os.path.join(output_dir, 'comsol_dualplot_2.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    
    print(f"\nPlot 2 saved to: {output_path2}")
    
    print(f"\n{'='*80}")
    print("✓ Visualization complete!")
    print(f"  - Total timesteps processed: {num_timesteps}")
    print(f"  - Total particles processed: {num_particles}")
    print(f"  - Plot 1: 3 time points")
    print(f"  - Plot 2: 3 time points")
    print(f"  - X-axis range: 2-32mm")
    print(f"{'='*80}\n")
    
    plt.close()


if __name__ == "__main__":
    main()
