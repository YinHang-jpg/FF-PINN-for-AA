import os
import csv
import time
import sys
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # 暂时注释掉，允许弹窗显示密度分布图
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, desc="", unit=""):
        return iterable

# 修复Windows中文编码问题
if sys.platform.startswith('win'):
    try:
        import codecs
        if hasattr(sys.stdout, 'detach'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        if hasattr(sys.stderr, 'detach'):
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except Exception:
        pass

# 导入初始化函数和DEM相关模块
from initialization.particle_initialization import initialize_particles
from initialization.sound_source_standing import frequency, sound_pressure_level
from mechanisms.ARF import compute_pressure_gradient_and_apply_arf
from mechanisms.Stokes_drag import apply_stokes_drag

# 物理参数（与 DEM_main.py 保持一致）
USE_STOKES_DRAG = True
USE_ARF_FORCES = False  # DEM_main.py中为False，关闭以提升速度
dt = 1e-6  # 时间步长 (s)
domain_size = (0.034, 0.034)  # 34mm x 34mm
steps = 10000  # 统一时步为10000
COLLISION_CHECK_INTERVAL = 10  # 每10步检查一次碰撞，减少计算量


def handle_collisions_1d(positions, velocities, radii, mass, particle_count):
    """
    一维碰撞检测：粒子水平分布，只考虑x方向
    """
    n = len(positions)
    if n < 2:
        return positions, velocities, radii, mass, particle_count, 0
    
    initial_count = n
    
    # 只提取x坐标和x方向速度
    x = positions[:, 0].copy()
    vx = velocities[:, 0].copy()
    
    # 按x坐标排序
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    vx = vx[sort_idx]
    radii = radii[sort_idx]
    mass = mass[sort_idx]
    particle_count = particle_count[sort_idx]
    
    # 检查相邻粒子碰撞（一次遍历）
    to_delete = []
    i = 0
    while i < n - 1:
        dx = x[i+1] - x[i]
        if dx < (radii[i] + radii[i+1]):
            # 碰撞：合并到粒子i
            m1, m2 = mass[i], mass[i+1]
            total_mass = m1 + m2
            
            x[i] = (x[i] * m1 + x[i+1] * m2) / total_mass
            vx[i] = (vx[i] * m1 + vx[i+1] * m2) / total_mass
            radii[i] = np.cbrt(radii[i]**3 + radii[i+1]**3)
            mass[i] = total_mass
            particle_count[i] += particle_count[i+1]
            
            to_delete.append(i+1)
            i += 2  # 跳过已合并的粒子
        else:
            i += 1
    
    # 删除被合并的粒子
    if to_delete:
        keep = np.ones(n, dtype=bool)
        keep[to_delete] = False
        x = x[keep]
        vx = vx[keep]
        radii = radii[keep]
        mass = mass[keep]
        particle_count = particle_count[keep]
        positions_new = positions[sort_idx][keep]
        velocities_new = velocities[sort_idx][keep]
    else:
        positions_new = positions[sort_idx]
        velocities_new = velocities[sort_idx]
    
    # 更新x坐标和x方向速度
    n_new = len(x)
    positions_new[:, 0] = x
    velocities_new[:, 0] = vx
    
    collisions_count = initial_count - n_new
    
    return positions_new, velocities_new, radii, mass, particle_count, collisions_count


def run_dem_simulation(particle_count: int):
    """
    运行一次DEM仿真，使用 DEM_main.py 的核心逻辑，添加碰撞检测
    返回: (total_collisions:int, elapsed_seconds:float, figure_path:str or None,
          initial_positions:np.ndarray, final_positions:np.ndarray, final_particle_counts:np.ndarray)
    """
    print(f"\n=== 开始DEM仿真，粒子数: {particle_count} ===")
    
    # 初始化粒子（与 DEM_main.py 一致）
    positions, velocities, radii, mass = initialize_particles(N=particle_count, domain_size=domain_size)
    initial_positions = positions.copy()
    initial_count = len(positions)
    
    # 初始化粒子计数数组（每个粒子初始计数为1）
    particle_count_np = np.ones(len(positions), dtype=np.int32)
    
    print(f"初始化了 {initial_count} 个粒子")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    print(f"开始计算: {steps} 步，时间步长: {dt:.2e} s")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    simulation_time = 0.0
    
    # 碰撞统计
    total_collisions = 0
    collisions_per_step = []
    
    # 主循环（与 DEM_main.py 的核心逻辑一致）
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time
        
        # 力学更新：计算各项力（与 DEM_main.py 一致）
        total_force = np.zeros_like(positions)
        
        if USE_ARF_FORCES:
            arf_force = compute_pressure_gradient_and_apply_arf(positions, radii, t)
            total_force += arf_force
        
        if USE_STOKES_DRAG:
            drag_force = apply_stokes_drag(positions, velocities, radii, mass, dt, 
                                         time=t, use_sound_velocity=True, 
                                         frequency=frequency, sound_pressure_level=sound_pressure_level)
            total_force += drag_force
        
        # 显式欧拉积分（与 DEM_main.py 一致）
        accelerations = total_force / mass[:, None]
        velocities = velocities + accelerations * dt
        positions = positions + velocities * dt
        
        # 碰撞检测（每隔COLLISION_CHECK_INTERVAL步检查一次，减少计算量）
        if (step + 1) % COLLISION_CHECK_INTERVAL == 0:
            positions, velocities, radii, mass, particle_count_np, collisions = handle_collisions_1d(
                positions, velocities, radii, mass, particle_count_np
            )
            total_collisions += collisions
            collisions_per_step.append(collisions)
        else:
            collisions_per_step.append(0)
        
        # 每10%进度打印一次
        if (step + 1) % (steps // 10) == 0:
            progress = (step + 1) / steps * 100
            print(f"进度: {step + 1}/{steps} ({progress:.1f}%) - 当前粒子数: {len(positions)}, 累计碰撞: {total_collisions}")
    
    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"总计算时间: {total_elapsed:.4f} s")
    print(f"初始粒子数: {initial_count}")
    print(f"最终粒子数: {len(positions)}")
    print(f"总碰撞次数: {total_collisions}")
    
    # 绘制四个子图的综合图
    figure_path = None
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        
        # 子图1：粒子初始化的空间分布
        ax1.scatter(initial_positions[:, 0] * 1000, initial_positions[:, 1] * 1000, 
                   s=1, c='blue', alpha=0.6, label='Initial Particles')
        ax1.set_xlim(0, domain_size[0] * 1000)
        ax1.set_ylim(0, domain_size[1] * 1000)
        ax1.set_xlabel('X Position (mm)', fontsize=11)
        ax1.set_ylabel('Y Position (mm)', fontsize=11)
        ax1.set_title(f'Initial Particle Distribution (N={initial_count})', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2：粒子计算结束后的空间分布
        ax2.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, 
                   s=1, c='red', alpha=0.6, label='Final Particles')
        ax2.set_xlim(0, domain_size[0] * 1000)
        ax2.set_ylim(0, domain_size[1] * 1000)
        ax2.set_xlabel('X Position (mm)', fontsize=11)
        ax2.set_ylabel('Y Position (mm)', fontsize=11)
        ax2.set_title(f'Final Particle Distribution (N={len(positions)})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 子图3：粒子的浓度分布曲线
        x_grid = np.linspace(0.0, 34.0, 100)
        
        def calculate_concentration_distribution(positions, x_grid_mm):
            if len(positions) == 0:
                return np.zeros_like(x_grid_mm)
            x_positions_mm = positions[:, 0] * 1000.0
            concentrations = np.zeros_like(x_grid_mm)
            sigma = 0.2
            for i, x_pos in enumerate(x_grid_mm):
                distances = np.abs(x_positions_mm - x_pos)
                concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
                concentrations[i] = concentration
            return concentrations
        
        initial_concentrations = calculate_concentration_distribution(initial_positions, x_grid)
        final_concentrations = calculate_concentration_distribution(positions, x_grid)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        ax3.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, 
               color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
        
        try:
            pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
            x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
            y_smooth = pchip(x_smooth)
            ax3.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
        except Exception:
            ax3.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
        
        ax3.set_xlim(0.0, 34.0)
        ax3.set_xlabel('x (mm)', fontsize=11)
        ax3.set_ylabel('Relative Change (%)', fontsize=11)
        ax3.set_title('Particle Density Distribution', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax3.set_ylim(y_min - y_margin, y_max + y_margin)
        ax3.legend()
        
        # 子图4：每个时步的碰撞次数折线图
        if len(collisions_per_step) > 0:
            steps_arr = np.arange(1, len(collisions_per_step) + 1, dtype=int)
            values = np.asarray(collisions_per_step, dtype=int)
            
            ax4.plot(steps_arr, values, '-', linewidth=1.5, color='crimson', alpha=0.7, label='Collisions per step')
            ax4.scatter(steps_arr, values, s=10, color='crimson', alpha=0.5, zorder=3)
            
            ax4.set_xlabel('Time Step', fontsize=11)
            ax4.set_ylabel('Collisions per Step', fontsize=11)
            ax4.set_title(f'Collision Trend (N={particle_count})', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No collision data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Collision Trend (N={particle_count})', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        dst_png = f'collision_trend_DEM_N_{particle_count}.png'
        plt.savefig(dst_png, dpi=300, bbox_inches='tight')
        plt.close()
        figure_path = os.path.abspath(dst_png)
        print(f"综合图已保存至: {figure_path}")
    except Exception as e:
        print(f"绘制综合图失败: {e}")
        import traceback
        traceback.print_exc()
        figure_path = None
    
    return total_collisions, total_elapsed, figure_path, initial_positions, positions, particle_count_np


def update_results_and_plot(results, last_simulation_data, csv_path='auto_test_results_DEM.csv', show_plot=False):
    """
    更新CSV结果并绘制图表（每次仿真完成后调用）
    """
    # 保存CSV
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['particles', 'total_collisions', 'elapsed_seconds', 'figure_path'])
            for n, c, t, p in results:
                writer.writerow([n, c, f'{t:.6f}', p or ''])
        print(f'[auto] CSV已更新: {len(results)} 组数据')
    except Exception as e:
        print(f'[auto] Failed to write CSV: {e}')
        return
    
    # 绘制耗时与粒子数目关系图和粒子浓度分布图
    try:
        particles = [n for n, _, _, _ in results]
        elapsed_times = [t for _, _, t, _ in results]
        
        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：耗时与粒子数目关系
        ax1.clear()
        ax1.plot(particles, elapsed_times, 's-', linewidth=2, markersize=8, color='#A23B72', label='Computation Time')
        ax1.set_xlabel('Number of Particles', fontsize=12)
        ax1.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax1.set_title('Computation Time vs Number of Particles (DEM)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 右图：粒子浓度分布（使用最后一个仿真结果）
        ax2.clear()
        if last_simulation_data is not None:
            initial_pos, final_pos, final_counts, n_particles = last_simulation_data
            
            # 创建x轴网格
            x_grid = np.linspace(0.0, 34.0, 100)
            
            # 计算浓度分布
            def calculate_concentration_distribution(positions, x_grid_mm):
                if len(positions) == 0:
                    return np.zeros_like(x_grid_mm)
                
                x_positions_mm = positions[:, 0] * 1000.0
                concentrations = np.zeros_like(x_grid_mm)
                sigma = 0.2
                
                for i, x_pos in enumerate(x_grid_mm):
                    distances = np.abs(x_positions_mm - x_pos)
                    concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
                    concentrations[i] = concentration
                
                return concentrations
            
            initial_concentrations = calculate_concentration_distribution(initial_pos, x_grid)
            final_concentrations = calculate_concentration_distribution(final_pos, x_grid)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
                relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
            
            ax2.set_xlim(0.0, 34.0)
            ax2.set_xlabel('x (mm)', fontsize=12)
            ax2.set_ylabel('Relative Change (%)', fontsize=12)
            ax2.set_title(f'Particle Density Distribution (N={n_particles})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            bars = ax2.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, 
                          color='skyblue', edgecolor='navy', linewidth=0.5, label='Concentration Change')
            
            try:
                pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
                x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
                y_smooth = pchip(x_smooth)
                ax2.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
            except Exception:
                ax2.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
            
            if np.any(np.isfinite(relative_change)):
                y_min = float(np.nanmin(relative_change))
                y_max = float(np.nanmax(relative_change))
                y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
                ax2.set_ylim(y_min - y_margin, y_max + y_margin)
            
            ax2.legend()
            print(f'[auto] 密度分布图已更新 (N={n_particles})')
        else:
            ax2.text(0.5, 0.5, 'No simulation data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Particle Density Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        time_plot_path = 'computation_time_vs_particle_count_DEM.png'
        plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
        print(f'[auto] 图表已保存: {time_plot_path}')
        
        # 不弹出图像，直接关闭
        plt.close()
    except Exception as e:
        print(f'[auto] Failed to plot: {e}')
        import traceback
        traceback.print_exc()


def main():
    # 循环粒子数：从1000到15000，步长1000
    particle_counts = list(range(1000, 15001, 1000))
    
    results = []  # 每项为 (N, collisions, elapsed_s, figure_path)
    last_simulation_data = None  # 保存最后一个仿真结果用于绘制浓度分布图
    csv_path = 'auto_test_results_DEM.csv'
    
    # 使用tqdm显示进度条
    for idx, n in enumerate(tqdm(particle_counts, desc="仿真进度", unit="组")):
        print(f"\n{'='*60}")
        print(f"实验 {idx+1}/{len(particle_counts)}: 粒子数 = {n}")
        print(f"{'='*60}")
        
        collisions, elapsed, figure_path, initial_pos, final_pos, final_counts = run_dem_simulation(n)
        results.append((n, collisions, elapsed, figure_path))
        last_simulation_data = (initial_pos, final_pos, final_counts, n)
        
        print(f'[auto] N={n}, collisions={collisions}, elapsed={elapsed:.3f}s, figure={figure_path or "N/A"}')
        
        # 每完成一组仿真就保存并更新结果（第一次显示图表）
        update_results_and_plot(results, last_simulation_data, csv_path, show_plot=(idx == 0))
        print(f'[auto] Results updated: {len(results)}/{len(particle_counts)} completed')
    
    print(f'\n[auto] 所有仿真完成！最终结果已保存至: {os.path.abspath(csv_path)}')
    
    # 最后保存一次完整的结果图（不显示）
    try:
        update_results_and_plot(results, last_simulation_data, csv_path, show_plot=False)
        print(f'[auto] 最终图表已保存')
    except Exception as e:
        print(f'[auto] Failed to save final plot: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
