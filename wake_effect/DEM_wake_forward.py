import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf
from mechanisms.Stokes_drag import apply_stokes_drag, compute_air_velocity_due_to_sound

try:
    from wake_effect.DEM_wake.DEM_wake import CHAR_LENGTH_M
except ImportError:
    from DEM_wake.DEM_wake import CHAR_LENGTH_M

# ============ 配置 ============
USE_ARF_FORCES = True
USE_STOKES_DRAG = True
USE_ACOUSTIC_WAKE = True
Re_FIXED = 1.0
N_PARTICLES = 1000
WAKE_CUTOFF_R = 30.0
domain_size = (0.034, 0.034)
dt = 1e-6
steps = 10000


def compute_wake_velocities(positions, velocities, Re):
    """
    向量化计算所有粒子对之间的尾流诱导速度。
    返回 (N, 2) 附加流体速度，用于修正 Stokes 阻力。
    """
    N = len(positions)
    rel = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
    x = rel[:, :, 0]
    y = rel[:, :, 1]

    r = np.hypot(x, y)

    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    mask &= (r > 1e-12)

    R = np.where(mask, r / CHAR_LENGTH_M, 1.0)
    mask &= (R < WAKE_CUTOFF_R)

    Theta = np.arctan2(y, x)
    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    cos_2t = np.cos(2 * Theta)

    # Oseen
    exp_arg = np.clip((-R * Re / 4) * (1 - cos_t), -500, 500)
    exp_term = np.exp(exp_arg)

    R_safe = np.where(mask, R, 1.0)
    R2 = R_safe ** 2
    R4 = R_safe ** 4

    Vr_Oseen = (cos_t / (2 * R_safe) -
                (3 * R_safe * (1 + cos_t) / 4) * exp_term +
                3 * (1 - exp_term) / Re) / R2

    Vt_Oseen = sin_t * (1 / R2 + 3 * exp_term) / (4 * R_safe)

    # PP
    PP_r1 = 4 * R_safe * (16 + 3 * Re + 3 * R2 * (-16 + (2 * R_safe - 3) * Re)) * cos_t
    PP_r2 = 3 * ((R_safe - 1) ** 2) * (1 + R_safe + 2 * R2) * (1 + 3 * cos_2t) * Re
    Vr_PP = (PP_r1 - PP_r2) / (128 * R4)

    PP_t1 = R_safe * (16 + 3 * Re + 3 * R2 * (16 + (3 - 4 * R_safe) * Re))
    PP_t2 = 3 * (-2 + R_safe - 3 * R_safe ** 3 + 4 * R4) * cos_t * Re
    Vt_PP = sin_t * (PP_t1 + PP_t2) / (64 * R4)

    # 分段混合
    pp_zone = R_safe < 2
    oseen_zone = R_safe > 5
    blend_zone = ~pp_zone & ~oseen_zone

    vr = np.where(pp_zone, Vr_PP,
         np.where(oseen_zone, Vr_Oseen,
                  ((5 - R_safe) * Vr_PP + (R_safe - 2) * Vr_Oseen) / 3))
    vt = np.where(pp_zone, Vt_PP,
         np.where(oseen_zone, Vt_Oseen,
                  ((5 - R_safe) * Vt_PP + (R_safe - 2) * Vt_Oseen) / 3))

    # 极坐标→笛卡尔
    U = vr * np.cos(Theta) - vt * np.sin(Theta)
    V = vr * np.sin(Theta) + vt * np.cos(Theta)

    # 方向调整（与 DEM_wake.py 保持一致）
    U_abs = np.abs(U)
    upstream = x > 0
    U = np.where(upstream, -U_abs, -U_abs * 0.5)

    # 屏蔽无效对并求和
    U = np.where(mask, U, 0.0)
    V = np.where(mask, V, 0.0)

    wake_vel = np.zeros((N, 2))
    wake_vel[:, 0] = np.sum(U, axis=1)
    wake_vel[:, 1] = np.sum(V, axis=1)
    return wake_vel


def main():
    global positions, velocities

    positions, velocities, radii, mass = initialize_particles(
        N=N_PARTICLES, domain_size=domain_size
    )
    initial_positions = positions.copy()
    simulation_time = 0.0

    print(f"DEM Wake Forward Simulation")
    print(f"粒子数: {N_PARTICLES}, 步数: {steps}, dt: {dt:.2e}")
    print(f"Wake: {USE_ACOUSTIC_WAKE}, ARF: {USE_ARF_FORCES}, Stokes: {USE_STOKES_DRAG}")

    total_start = time.perf_counter()

    for step in range(steps):
        simulation_time += dt
        t = simulation_time

        total_force = np.zeros_like(positions)

        if USE_ARF_FORCES:
            arf_force = compute_pressure_gradient_and_apply_arf(positions, radii, t)
            total_force += arf_force

        wake_fluid_vel = None
        if USE_ACOUSTIC_WAKE:
            wake_fluid_vel = compute_wake_velocities(positions, velocities, Re_FIXED)

        if USE_STOKES_DRAG:
            base_fluid_vel = compute_air_velocity_due_to_sound(
                positions, t, frequency, sound_pressure_level
            )
            if wake_fluid_vel is not None:
                total_fluid_vel = base_fluid_vel + wake_fluid_vel
            else:
                total_fluid_vel = base_fluid_vel

            drag_force = apply_stokes_drag(
                positions, velocities, radii, mass, dt,
                fluid_velocity=total_fluid_vel,
                time=t, use_sound_velocity=False,
                frequency=frequency, sound_pressure_level=sound_pressure_level
            )
            total_force += drag_force

        accelerations = total_force / mass[:, None]
        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt

        progress_interval = max(1, steps // 10)
        if (step + 1) % progress_interval == 0:
            elapsed = time.perf_counter() - total_start
            pct = (step + 1) / steps * 100
            print(f"进度: {step+1}/{steps} ({pct:.1f}%) - 已用时: {elapsed:.2f}s")

        if step == steps - 1:
            final_positions = positions.copy()

    total_elapsed = time.perf_counter() - total_start
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"最终仿真时间: {simulation_time:.6f} s")

    # ============ 绘图 ============
    x_grid = np.linspace(0.0, 34.0, 100)

    def calc_concentration(pos):
        x_mm = pos[:, 0] * 1000.0
        x_mm = x_mm[np.isfinite(x_mm)]
        c = np.zeros_like(x_grid)
        for i, xp in enumerate(x_grid):
            c[i] = np.sum(np.abs(x_mm - xp) <= 0.2)
        return c

    init_conc = calc_concentration(initial_positions)
    final_conc = calc_concentration(final_positions)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = (final_conc - init_conc) / init_conc * 100.0
        rel_change = np.nan_to_num(rel_change, nan=0.0, posinf=0.0, neginf=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000,
                s=(radii * 1e6 * 4) ** 2, c='blue', alpha=0.6)
    ax1.set_xlim(0, domain_size[0] * 1000)
    ax1.set_ylim(0, domain_size[1] * 1000)
    ax1.set_title("Particle Positions (with Wake Effect)")
    ax1.set_xlabel("x (mm)")
    ax1.set_ylabel("y (mm)")

    ax2.bar(x_grid, rel_change, width=0.34, align='center', alpha=0.8)
    try:
        pchip = PchipInterpolator(x_grid, rel_change, extrapolate=False)
        xs = np.linspace(x_grid[0], x_grid[-1], 400)
        ax2.plot(xs, pchip(xs), 'r-', linewidth=2, label='Smoothed')
    except Exception:
        pass
    ax2.set_title("Particle Density Change Rate (with Wake Effect)")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("Concentration Change (%)")
    ax2.grid(True, alpha=0.3)
    if np.any(np.isfinite(rel_change)):
        y_min = float(np.nanmin(rel_change))
        y_max = float(np.nanmax(rel_change))
        y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)

    plt.tight_layout()
    plt.show()

    out_path = os.path.join(PROJECT_ROOT, 'density_curve_DEM_wake.txt')
    np.savetxt(out_path, np.vstack([x_grid, rel_change]).T)
    print(f"密度曲线已保存至 {out_path}")


if __name__ == "__main__":
    main()
