import matplotlib.pyplot as plt

import matplotlib

import matplotlib.animation as animation
from initialization.particle_initialization import initialize_particles
import numpy as np
import psutil
import time
import threading
import os
import torch
import json
import sys


# Fix Windows console UTF-8
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator

RUN_HEADLESS_BENCHMARK = True  # Headless benchmark: no rendering; exit after 1000 steps
USE_ARF_FORCES = False  # Use analytical SPGF/ARF for all particles
USE_GRAVITY = False
USE_STOKES_DRAG = True  # Enable Stokes drag
USE_AGGLOMERATION = False
USE_BROWNIAN = False
USE_COLLISION = False  # Disable collision handling

# Debug print settings
TARGET_PARTICLE_INDEX = 0  # Particle index to monitor


# Simulation time accumulators
simulation_time = 0.0  # Accumulated simulation time
density_plot_saved = False  # Whether a density plot has been saved
last_save_time = 0.0  # Last density-plot save time

# Performance timing flags
step_timing_printed = False  # Whether step-5 timing has been printed



# Import acoustic field (visualization only)
from initialization.sound_source_standing import compute_sound_field, frequency, sound_pressure_level, spl_to_pressure
IS_STANDING_WAVE = True

def get_amplitude():
    return 2 * spl_to_pressure(sound_pressure_level)

# Import analytical SPGF/ARF module
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf

# Import Stokes drag module
from mechanisms.Stokes_drag import apply_stokes_drag, cunningham_correction_factor

# Print helper: Stokes drag from velocity (matches mechanisms/Stokes_drag.py)
def compute_stokes_drag_force(positions, velocities, radii, viscosity=1.79e-5, fluid_velocity=None, lambda_g=6.5e-8, time=0.0, use_sound_velocity=True):
    N = len(positions)
    if fluid_velocity is None:
        if use_sound_velocity:
            # Use acoustically driven air velocity
            from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
            fluid_velocity = compute_air_velocity_due_to_sound(positions, time)
        else:
            fluid_velocity = np.zeros((N, 2))
    drag_force = np.zeros_like(positions)
    for i in range(N):
        d_p = 2.0 * radii[i]
        C_c = cunningham_correction_factor(d_p, lambda_g)
        relative_velocity = velocities[i] - fluid_velocity[i]
        drag_coeff = 3.0 * np.pi * viscosity * d_p / C_c
        drag_force[i] = -drag_coeff * relative_velocity
    return drag_force


# Performance monitoring state
performance_data = {
    'cpu_percent': 0.0,
    'memory_percent': 0.0,
    'memory_used_gb': 0.0,
    'fps': 0.0,
    'frame_time': 0.0,
    'particle_count': 0
}

# Performance monitoring helpers
def update_performance_data():
    """Update performance metrics"""
    while True:
        try:
            # CPU utilization
            performance_data['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # Memory utilization and usage
            memory = psutil.virtual_memory()
            performance_data['memory_percent'] = memory.percent
            performance_data['memory_used_gb'] = memory.used / (1024**3)  # Convert to GB
            
            # Particle count
            performance_data['particle_count'] = len(positions) if 'positions' in globals() else 0
            
            time.sleep(0.5)  # Update every 0.5 s
        except:
            pass

# Start performance monitoring thread
performance_thread = threading.Thread(target=update_performance_data, daemon=True)
performance_thread.start()



# Initialize particles
domain_size = (0.034, 0.034)
positions, velocities, radii, mass = initialize_particles(N=10000, domain_size=domain_size)
# Store initial positions for displacement metrics
initial_positions = positions.copy()

#
print("Initializing analytical SPGF/ARF...")
try:
    print("Analytical SPGF/ARF ready")
    print(f"Acoustic parameters: frequency={frequency}Hz, amplitude={get_amplitude()*1000:.3f}mm")
    print(f"Domain size: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    
except Exception as e:
    print(f"Failed to initialize analytical SPGF/ARF: {e}")
    print("Particle force evaluation will be unavailable")

#

if RUN_HEADLESS_BENCHMARK:
    #
    steps = 40000
    dt = 1e-6
    print(f"Headless benchmark started: steps={steps}, dt={dt}")
    print("...")
    
    #
    comsol_edges_bench = np.array([-14.875, -10.625, -6.375, -2.125, 2.125, 6.375, 10.625, 14.875])
    edges_bench = comsol_edges_bench + 17.0  # DEM
    num_bins_bench = len(edges_bench) - 1  # 7
    M_total_bench = int(positions.shape[0] * 0.875)  # 87.5%
    
    #
    total_start_time = time.perf_counter()
    bench_t0 = time.perf_counter()
    
    for step in range(steps):
        #
        simulation_time += dt
        t = simulation_time

        #
        total_force = np.zeros_like(positions)
        if USE_ARF_FORCES:
            arf_force = compute_pressure_gradient_and_apply_arf(
                positions, radii, t
            )

            total_force += arf_force

        if USE_STOKES_DRAG:
            drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                         time=t, use_sound_velocity=True, 
                                         frequency=frequency, sound_pressure_level=sound_pressure_level)
            total_force += drag_force

        #
        accelerations = total_force / mass[:, None]
        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt

        #
        if step == 99:  # 100(0)
            print(f"\n=== 100 (t = {t:.6f} s) ===")
            
            #
            print(f" (m/s):")
            for i in range(min(10, len(velocities))):  # 10
                print(f"   {i}: vx = {velocities[i, 0]:.6e}, vy = {velocities[i, 1]:.6e}")
            if len(velocities) > 10:
                print(f"  ... ( {len(velocities) - 10} )")
            
            #
            from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
            air_velocity = compute_air_velocity_due_to_sound(
                positions, t, frequency, sound_pressure_level
            )
            print(f"\n (m/s):")
            for i in range(min(10, len(air_velocity))):  # 10
                print(f"   {i}: ux = {air_velocity[i, 0]:.6e}, uy = {air_velocity[i, 1]:.6e}")
            if len(air_velocity) > 10:
                print(f"  ... ( {len(air_velocity) - 10} )")
            
            #
            if USE_ARF_FORCES:
                arf_force = compute_pressure_gradient_and_apply_arf(positions, radii, t)
                print(f"\n/ARF (N):")
                for i in range(min(10, len(arf_force))):  # 10
                    print(f"   {i}: Fx = {arf_force[i, 0]:.6e}, Fy = {arf_force[i, 1]:.6e}")
                if len(arf_force) > 10:
                    print(f"  ... ( {len(arf_force) - 10} )")
            else:
                print(f"\n/ARF: ")
            
            #
            if USE_STOKES_DRAG:
                drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                             time=t, use_sound_velocity=True, 
                                             frequency=frequency, sound_pressure_level=sound_pressure_level)
                print(f"\n (N):")
                for i in range(min(10, len(drag_force))):  # 10
                    print(f"   {i}: Fx = {drag_force[i, 0]:.6e}, Fy = {drag_force[i, 1]:.6e}")
                if len(drag_force) > 10:
                    print(f"  ... ( {len(drag_force) - 10} )")
            else:
                print(f"\n: ")
            
            #
            print(f"\n (N):")
            for i in range(min(10, len(total_force))):  # 10
                print(f"   {i}: Fx = {total_force[i, 0]:.6e}, Fy = {total_force[i, 1]:.6e}")
            if len(total_force) > 10:
                print(f"  ... ( {len(total_force) - 10} )")
            
            #
            relative_velocity = velocities - air_velocity
            print(f"\n ( - ) (m/s):")
            for i in range(min(10, len(relative_velocity))):  # 10
                print(f"   {i}: vx_rel = {relative_velocity[i, 0]:.6e}, vy_rel = {relative_velocity[i, 1]:.6e}")
            if len(relative_velocity) > 10:
                print(f"  ... ( {len(relative_velocity) - 10} )")
            
            print(f"=== 100 ===\n")

        #
        progress_interval = max(1, steps // 10)  # 10%
        if (step + 1) % progress_interval == 0:
            current_time = time.perf_counter()
            elapsed_so_far = current_time - total_start_time
            progress = (step + 1) / steps * 100
            print(f": {step + 1}/{steps} ({progress:.1f}%) - : {elapsed_so_far:.2f}s")
        if step == steps-1:
            #
            final_positions = positions.copy()
    #
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    bench_t1 = time.perf_counter()
    physics_elapsed = bench_t1 - bench_t0
    
    print(f"\n===  ===")
    print(f": {steps}")
    print(f": {dt:.2e} s")
    print(f": {total_elapsed:.4f} s ({total_elapsed/60:.2f} )")
    print(f": {total_elapsed/steps*1e3:.3f} ms")
    print(f": {physics_elapsed:.4f} s")
    print(f": {simulation_time:.6f} s")


    #
    def _optimize_bandwidth_1d(x_coords_mm, max_samples_for_cv=50):
        """KDE(Scott,)."""
        x = np.asarray(x_coords_mm)
        x = x[np.isfinite(x)]
        if x.size < 2:
            return None
        #
        idx = np.arange(x.size)
        if x.size > max_samples_for_cv:
            rng = np.random.default_rng(42)
            cv_idx = rng.choice(idx, size=max_samples_for_cv, replace=False)
        else:
            cv_idx = idx

        def objective(factor):
            if factor <= 0:
                return np.inf
            log_like = 0.0
            for i in cv_idx:
                train = np.delete(x, i)
                try:
                    kde = gaussian_kde(train, bw_method=float(factor))
                    pdf_val = float(kde(x[i])[0])
                    if pdf_val > 0:
                        log_like += np.log(pdf_val)
                except Exception:
                    return np.inf
            return -log_like

        #
        res = minimize_scalar(objective, bounds=(0.05, 1.0), method='bounded')
        return float(res.x) if res.success else 1.0

    def _kde_density_1d(x_coords_mm, eval_grid_mm, bw=None):
        x = np.asarray(x_coords_mm)
        if x.size == 0:
            return np.zeros_like(eval_grid_mm)
        if bw is None:
            bw = _optimize_bandwidth_1d(x)
        kde = gaussian_kde(x, bw_method=float(bw))
        return kde(eval_grid_mm)

    def _smooth_preserve_minmax(_y, window_ratio_base=0.07, window_ratio_strong=0.15, ptp_flat_threshold=0.5, min_limit=-99.9):
        """Hanning:
        - (- < ):,;
        - :[min, max];
        - ,;-100%.
        :_y(%),ptp_flat_threshold.
        """
        y = np.asarray(_y)
        n = y.size
        if n < 5:
            return np.maximum(y, min_limit)

        y_min = float(np.min(y))
        y_max = float(np.max(y))
        ptp = y_max - y_min

        #
        is_flat = bool(ptp <= float(ptp_flat_threshold))
        window_ratio = float(window_ratio_strong if is_flat else window_ratio_base)

        w = max(5, int(n * window_ratio))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        if w < 5:
            return np.maximum(y, min_limit)

        pad = w // 2
        y_pad = np.pad(y, (pad, pad), mode='reflect')
        kernel = np.hanning(w)
        kernel = kernel / np.sum(kernel)
        y_s = np.convolve(y_pad, kernel, mode='valid')

        if not is_flat:
            ys_min = float(np.min(y_s))
            ys_max = float(np.max(y_s))
            if ys_max > ys_min and y_max > y_min:
                a = (y_max - y_min) / (ys_max - ys_min)
                b = y_min - a * ys_min
                y_s = a * y_s + b

        y_s = np.maximum(y_s, min_limit)
        return y_s

    fig_h, (ax_p_h, ax_d_h) = plt.subplots(1, 2, figsize=(16, 6))

    #
    ax_p_h.set_xlim(0, domain_size[0] * 1000)
    ax_p_h.set_ylim(0, domain_size[1] * 1000)
    ax_p_h.set_title("Particle Positions (after computation)")
    ax_p_h.set_xlabel("x (mm)")
    ax_p_h.set_ylabel("y (mm)")
    ax_p_h.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, s=(radii * 1e6 * 4)**2, c='blue', alpha=0.6)

    #
    #
    x_grid_h = np.linspace(0.0, 34.0, 100)  # 100
    
    def _calculate_concentration_distribution_h(positions):
        """ - ,"""
        try:
            #
            current_positions = positions[:, 0] * 1000.0  # x
            current_positions = current_positions[np.isfinite(current_positions)]
            
            if len(current_positions) == 0:
                return np.zeros_like(x_grid_h)
            
            #
            concentrations = np.zeros_like(x_grid_h)
            
            #
            window_size = 0.2  # 0.5mm
            
            for i, x_pos in enumerate(x_grid_h):
                #
                #
                in_window = np.abs(current_positions - x_pos) <= window_size
                concentration = np.sum(in_window)
                
                concentrations[i] = concentration
            
            return concentrations
        except Exception as e:
            print(f": {e}")
            return np.zeros_like(x_grid_h)
    
    #
    initial_concentrations_h = _calculate_concentration_distribution_h(initial_positions)
    final_concentrations_h = _calculate_concentration_distribution_h(final_positions)
    
    #
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change_h = (final_concentrations_h - initial_concentrations_h) / initial_concentrations_h * 100.0
        relative_change_h = np.nan_to_num(relative_change_h, nan=0.0, posinf=0.0, neginf=0.0)
    
    bars_h = ax_d_h.bar(x_grid_h, relative_change_h, width=0.34, align='center', alpha=0.8)
    #
    try:
        pchip_h = PchipInterpolator(x_grid_h, relative_change_h, extrapolate=False)
        x_smooth_h = np.linspace(x_grid_h[0], x_grid_h[-1], max(50, 4 * len(x_grid_h)))
        y_smooth_h = pchip_h(x_smooth_h)
        curve_line_h, = ax_d_h.plot(x_smooth_h, y_smooth_h, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        curve_line_h = None
    ax_d_h.set_xlim(x_grid_h[0], x_grid_h[-1])
    if np.any(np.isfinite(relative_change_h)):
        y_min_h = float(np.nanmin(relative_change_h))
        y_max_h = float(np.nanmax(relative_change_h))
        y_margin_h = (y_max_h - y_min_h) * 0.1 if y_max_h > y_min_h else 1.0
        ax_d_h.set_ylim(y_min_h - y_margin_h, y_max_h + y_margin_h)
    ax_d_h.set_title("Particle Density Change Rate (after computation)")
    ax_d_h.set_xlabel("x (mm)")
    ax_d_h.set_ylabel("Concentration Change (%)")
    ax_d_h.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    #
    np.savetxt('density_curve_DEM.txt', np.vstack([x_grid_h, relative_change_h]).T)

    sys.exit(0)

#
fig, (ax_particles, ax_density) = plt.subplots(1, 2, figsize=(16, 6))

#
ax_particles.set_xlim(0, domain_size[0] * 1000)
ax_particles.set_ylim(0, domain_size[1] * 1000)
ax_particles.set_title("Particle Motion")
ax_particles.set_xlabel("x (mm)")
ax_particles.set_ylabel("y (mm)")

#
ax_density.set_xlim(0.0, 34.0)  # x_grid
ax_density.set_ylim(-5, 5)  # y,
ax_density.set_title("Particle Density Distribution")
ax_density.set_xlabel("x (mm)")
ax_density.set_ylabel("Concentration Change (%)")
ax_density.grid(True, alpha=0.3)

#

#
X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=0.0)
sound_img = ax_particles.imshow(
    P,
    extent=(0, domain_size[0] * 1000, 0, domain_size[1] * 1000),
    origin='lower',
    cmap='RdBu_r',
    alpha=0.4,
    vmin=-get_amplitude(),
    vmax=get_amplitude()
)

#
scatter = ax_particles.scatter(positions[:, 0] * 1000,
                     positions[:, 1] * 1000,
                     s=(radii * 1e6 * 4)**2,
                     c='blue', alpha=0.6)

#

#
performance_text = ax_particles.text(0.02, 0.98, '', transform=ax_particles.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

"""(comsol_visualizer.py)"""
#
x_grid = np.linspace(0.0, 34.0, 100)  # 100
concentration_values = np.zeros_like(x_grid)

#
concentration_scale = 100.0  # 
distance_weight = 1.0  # 
baseline_concentration = 0.0  # ()

#
def _calculate_concentration_distribution(positions):
    """ - ,"""
    try:
        #
        current_positions = positions[:, 0] * 1000.0  # x
        current_positions = current_positions[np.isfinite(current_positions)]
        
        if len(current_positions) == 0:
            return np.zeros_like(x_grid)
        
        #
        concentrations = np.zeros_like(x_grid)
        
        #
        window_size = 0.5  # 0.5mm
        
        for i, x_pos in enumerate(x_grid):
            #
            #
            in_window = np.abs(current_positions - x_pos) <= window_size
            concentration = np.sum(in_window)
            
            concentrations[i] = concentration
        
        return concentrations
    except Exception as e:
        print(f": {e}")
        return np.zeros_like(x_grid)

#
initial_concentrations = _calculate_concentration_distribution(positions)
baseline_concentration = np.mean(initial_concentrations)

#
print(f"\n[] :")
print(f"  x_grid: {x_grid[0]:.1f}mm - {x_grid[-1]:.1f}mm")
print(f"  : {concentration_scale}")
print(f"  : {distance_weight}")
print(f"  : {baseline_concentration:.3f}")
print(f"  : {len(positions)}")

#
with np.errstate(divide='ignore', invalid='ignore'):
    relative_change0 = (initial_concentrations - initial_concentrations) / initial_concentrations * 100.0
    relative_change0 = np.nan_to_num(relative_change0, nan=0.0, posinf=0.0, neginf=0.0)

#
bars = ax_density.bar(x_grid, relative_change0, width=0.34, align='center', alpha=0.8, label='Concentration Change')
#
try:
    pchip0 = PchipInterpolator(x_grid, relative_change0, extrapolate=False)
    x_smooth0 = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
    y_smooth0 = pchip0(x_smooth0)
    curve_line, = ax_density.plot(x_smooth0, y_smooth0, 'r-', linewidth=2, label='Smoothed')
except Exception:
    curve_line = ax_density.plot([], [], 'r-', linewidth=2, label='Smoothed')[0]
ax_density.set_xlim(x_grid[0], x_grid[-1])
if np.any(np.isfinite(relative_change0)):
    y_min0 = float(np.nanmin(relative_change0))
    y_max0 = float(np.nanmax(relative_change0))
    y_margin0 = (y_max0 - y_min0) * 0.1 if y_max0 > y_min0 else 1.0
    ax_density.set_ylim(y_min0 - y_margin0, y_max0 + y_margin0)
ax_density.legend()



#
frame_times = []
last_frame_time = time.time()

def update(frame):
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, step_timing_printed

    #
    if frame < 5:  # 5
        print(f"Frame {frame}: Starting update function")

    #
    render_interval = 1  # ,
    should_render = (frame % render_interval == 0)

    #
    if frame == 5 and not step_timing_printed:
        t0 = time.perf_counter()

    #
    current_time = time.time()
    frame_time = current_time - last_frame_time
    frame_times.append(frame_time)
    
    #
    if len(frame_times) > 30:
        frame_times.pop(0)
    
    #
    if len(frame_times) > 0:
        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        fps = 0
    
    performance_data['fps'] = fps
    performance_data['frame_time'] = frame_time * 1000  # 
    
    last_frame_time = current_time

    dt = 1e-8
    global simulation_time
    simulation_time += dt  # Accumulated simulation time
    t = simulation_time  # 

    #
    if frame == 5 and not step_timing_printed:
        t1 = time.perf_counter()

    #
    total_force = np.zeros_like(positions)
    if USE_ARF_FORCES:
        if frame < 5:
            print(f"Frame {frame}: Computing ARF forces")
        try:
            arf_force = compute_pressure_gradient_and_apply_arf(
                positions, radii, t
            )
            print(t)
            total_force += arf_force
            if frame < 5:
                print(f"Frame {frame}: ARF forces computed successfully")
        except Exception as e:
            print(f"Frame {frame}: ARF calculation failed: {e}")
            import traceback
            traceback.print_exc()
    
    #
    if frame == 5 and not step_timing_printed:
        t2 = time.perf_counter()
    
    #
    if USE_STOKES_DRAG:
        drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                     time=t, use_sound_velocity=True, 
                                     frequency=frequency, sound_pressure_level=sound_pressure_level)
        total_force += drag_force

    #
    accelerations = total_force / mass[:, None]
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt
    
    #
    if frame == 5 and not step_timing_printed:
        t3 = time.perf_counter()

    #
    if should_render:
        Nx, Ny = (200, 200)
        _, _, P = compute_sound_field(domain_size=domain_size, resolution=(Nx, Ny), time=t)
        sound_img.set_data(P)
        scatter.set_offsets(positions * 1000)
    
    #
    if frame == 5 and not step_timing_printed:
        t4 = time.perf_counter()
    
    if should_render:
        #
        current_concentrations = _calculate_concentration_distribution(positions)
        
        #
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (current_concentrations - initial_concentrations) / initial_concentrations * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        #
        for rect, h in zip(bars, relative_change):
            rect.set_height(h)

        #
        try:
            pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
            x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
            y_smooth = pchip(x_smooth)
            curve_line.set_data(x_smooth, y_smooth)
        except Exception:
            curve_line.set_data(x_grid, relative_change)

        #
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax_density.set_ylim(y_min - y_margin, y_max + y_margin)

        ax_density.set_title(f"Particle Density Change Rate (t = {t:.6f} s)")
    
    #
    if frame == 5 and not step_timing_printed:
        t5 = time.perf_counter()


    
    #
    if should_render:
        if t >= 0.01 and t <= 0.2 and (t - last_save_time) >= 0.01:
            #
            plt.figure(figsize=(12, 8))
            plt.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, label='Concentration Change (%)')
            plt.xlabel('x (mm)')
            plt.ylabel('Concentration Change (%)')
            plt.title(f'Particle Density Change Rate at t = {t:.6f} s')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            #
            filename = f'density_plot_{int(t*1000):03d}_t_{t:.6f}s.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            last_save_time = t  # 

    #
    arf_status = "✓ ARF Active" if USE_ARF_FORCES else "✗ ARF Inactive"
    stokes_status = "✓ Stokes Drag Active" if USE_STOKES_DRAG else "✗ Stokes Drag Inactive"
    if should_render:
        performance_text.set_text(
            f'Simulation Time: {t:.6f} s\n'
            f'FPS: {performance_data["fps"]:.1f}\n'
            f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
            f'Particles: {performance_data["particle_count"]}\n'
            f'ARF Physics: {arf_status}\n'
            f'Stokes Drag: {stokes_status}'
        )
    
    #
    if frame == 5 and not step_timing_printed:
        t6 = time.perf_counter()
    
    #
    if frame == 5 and not step_timing_printed:
        pending_draw_measurement = {'draw_start': time.perf_counter(), 'draw_end': None}
        fig.canvas.draw()
        if pending_draw_measurement['draw_end'] is None:
            #
            pending_draw_measurement['draw_end'] = time.perf_counter()

        def ms(x):
            return max(0.0, x) * 1000.0

        #
        t1_l = locals().get('t1', t0)
        t2_l = locals().get('t2', t1_l)
        t3_l = locals().get('t3', t2_l)
        t4_l = locals().get('t4', t3_l)
        t5_l = locals().get('t5', t4_l)
        t6_l = locals().get('t6', t5_l)

        t_sound = ms(t1_l - t0)
        t_arf = ms(t2_l - t1_l)
        t_drag = ms(t3_l - t2_l)
        t_scatter = ms(t4_l - t3_l)
        t_density = ms(t5_l - t4_l)
        t_perf = ms(t6_l - t5_l)
        draw_ms = ms(pending_draw_measurement['draw_end'] - pending_draw_measurement['draw_start'])
        total_ms = t_sound + t_arf + t_drag + t_scatter + t_density + t_perf + draw_ms

        print("\nStep timing breakdown (frame 5):")
        print(f"  Particles:          {len(positions)}")
        print(f"  Sound field update: {t_sound:.2f} ms")
        print(f"  ARF computation:    {t_arf:.2f} ms")
        print(f"  Stokes drag:        {t_drag:.2f} ms")
        print(f"  Scatter update:     {t_scatter:.2f} ms")
        print(f"  Density calculation:{t_density:.2f} ms")
        print(f"  Perf display:       {t_perf:.2f} ms")
        print(f"  Render (draw):      {draw_ms:.2f} ms")
        print(f"  Total (incl. draw): {total_ms:.2f} ms")
        step_timing_printed = True
    
    return sound_img, scatter, performance_text, bars


ani = animation.FuncAnimation(fig, update, frames=20000, interval=50, blit=False)  # 50ms,20fps
plt.show()
