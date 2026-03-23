# 修复Windows中文编码问题
import sys
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

"""
COMSOL粒子位置数据可视化工具

功能：
    1. 读取CSV格式的粒子位置数据
    2. 使用高斯核密度估计计算粒子数密度分布
    3. 生成两张图：
       - 图1: 绝对密度演化（各时刻的粒子密度分布）
       - 图2: 相对密度变化（相对于初始时刻的密度变化百分比）

CSV数据格式要求：
    - 第一列：时间（单位：秒）
    - 后续各列：每个粒子在x轴上的位置（单位：毫米，范围0-34mm）
    - 行数：任意（每行对应一个时刻）
    - 列数：任意（第一列是时间，后续列是粒子位置）

示例数据结构：
    time,particle_1,particle_2,particle_3,...
    0.000,5.2,8.1,15.3,...
    0.001,5.3,8.2,15.1,...
    0.002,5.5,8.3,14.9,...
    ...

使用方法：
    python validation/comsol_visualizer.py

输出：
    - validation/comsol_density_evolution.png  (密度演化图)
    - validation/comsol_density_change.png     (相对变化图)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

def calculate_particle_density(particle_positions_mm, x_grid_mm, bandwidth=0.5):
    """
    使用高斯核密度估计计算粒子数密度
    
    参数:
        particle_positions_mm: 一维数组，包含所有粒子的x坐标（单位：mm）
        x_grid_mm: 一维数组，用于计算密度的x轴网格点（单位：mm）
        bandwidth: 高斯核的带宽（单位：mm），控制平滑程度
    
    返回:
        density: 在x_grid_mm各点的粒子数密度
    
    说明:
        - 使用高斯核：exp(-distance²/(2*bandwidth²))
        - 自动过滤NaN和超出范围的值
        - 适用于任意数量的粒子
    """
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    # 移除无效值（NaN或超出范围的值）
    particle_positions_mm = particle_positions_mm[~np.isnan(particle_positions_mm)]
    particle_positions_mm = particle_positions_mm[(particle_positions_mm >= 0) & (particle_positions_mm <= 34)]
    
    if len(particle_positions_mm) == 0:
        return np.zeros_like(x_grid_mm)
    
    density = np.zeros_like(x_grid_mm)
    
    # 高斯核密度估计
    for idx, x_pos in enumerate(x_grid_mm):
        # 计算所有粒子到当前位置的距离
        distances = np.abs(particle_positions_mm - x_pos)
        
        # 使用高斯核计算密度贡献
        density_contribution = np.exp(-distances**2 / (2 * bandwidth**2))
        density[idx] = np.sum(density_contribution)
    
    return density


def main():
    """
    主函数：读取COMSOL数据并绘制粒子数密度随时间的变化
    
    功能：
    - 自动检测CSV文件中的时间点数量（第一列）
    - 自动处理任意数量的粒子（后续列）
    - 生成两张图：绝对密度演化图 + 相对密度变化图
    """
    
    print("="*80)
    print("COMSOL粒子位置数据可视化 (支持任意时间点数量)")
    print("="*80)
    
    # 读取CSV文件
    csv_path = 'validation/comsol_positions.csv'
    
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return
    
    print(f"\n正在读取文件: {csv_path}")
    # 使用header=None，将所有行作为数据读取（不把第一行当作列名）
    df = pd.read_csv(csv_path, header=None)
    
    # 自动检测时间点数量和粒子数量
    num_timesteps = len(df)
    num_particles = len(df.columns) - 1
    
    print(f"数据维度: {df.shape}")
    print(f"✓ 检测到 {num_timesteps} 个时间点")
    print(f"✓ 检测到 {num_particles} 个粒子")
    
    # 验证数据有效性
    if num_timesteps == 0:
        print("错误：CSV文件中没有数据行")
        return
    if num_particles == 0:
        print("错误：CSV文件中没有粒子位置列（除时间列外）")
        return
    
    # 第一列是时间
    time_values = df.iloc[:, 0].values
    
    # 检查时间列中是否有NaN值
    nan_count = np.sum(np.isnan(time_values))
    if nan_count > 0:
        print(f"\n⚠️  警告：时间列中检测到 {nan_count} 个NaN值！")
        print(f"   有效时间点数量: {num_timesteps - nan_count}")
        print(f"   无效时间点位置: {np.where(np.isnan(time_values))[0].tolist()}")
        
        # 过滤掉NaN时间点
        valid_indices = ~np.isnan(time_values)
        time_values_valid = time_values[valid_indices]
        particle_positions_valid = df.iloc[:, 1:].values[valid_indices, :]
        
        print(f"\n   已自动过滤掉无效时间点，继续处理 {len(time_values_valid)} 个有效时间点")
        
        # 更新数据
        time_values = time_values_valid
        particle_positions = particle_positions_valid
        num_timesteps = len(time_values)
        
        if num_timesteps == 0:
            print("\n错误：过滤后没有有效的时间点！")
            return
    else:
        # 没有NaN值，正常读取粒子位置
        particle_positions = df.iloc[:, 1:].values  # shape: (n_timesteps, n_particles)
    
    print(f"\n时间范围: {time_values[0]:.6f}s 到 {time_values[-1]:.6f}s")
    
    # 根据时间点数量决定是否打印所有时间点
    if num_timesteps <= 20:
        print(f"时间点: {time_values}")
    else:
        print(f"时间点（前5个）: {time_values[:5]}")
        print(f"时间点（后5个）: {time_values[-5:]}")
    
    # 定义x轴网格（用于计算密度分布）
    x_grid_mm = np.linspace(0, 34, 200)  # 从0到34mm，200个点
    
    # 设置绘图参数
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 根据时间点数量动态生成颜色映射（从蓝色到红色）
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_timesteps))
    
    print(f"\n{'='*80}")
    print(f"开始计算各时刻的粒子密度分布（共{num_timesteps}个时刻）...")
    print(f"{'='*80}\n")
    
    # 存储所有密度曲线用于后续分析
    all_densities = []
    
    # 对每个时间点计算并绘制密度分布
    for i, t in enumerate(time_values):
        # 获取当前时刻所有粒子的位置
        positions_at_t = particle_positions[i, :]
        
        # 计算粒子数密度
        density = calculate_particle_density(positions_at_t, x_grid_mm, bandwidth=0.5)
        all_densities.append(density)
        
        # 对密度曲线进行平滑处理（可选）
        density_smoothed = gaussian_filter1d(density, sigma=2)
        
        # 绘制密度曲线
        label = f't = {t:.4f}s ({t*1000:.2f}ms)'
        ax.plot(x_grid_mm, density_smoothed, 
                color=colors[i], 
                linewidth=2.0, 
                alpha=0.8,
                label=label)
        
        print(f"时刻 {i+1}/{len(time_values)}: t={t:.6f}s, "
              f"有效粒子数={np.sum(~np.isnan(positions_at_t))}, "
              f"最大密度={np.max(density):.2f}")
    
    # 设置图表属性
    ax.set_xlim(0, 34)
    ax.set_xlabel('位置 x (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('粒子数密度 (任意单位)', fontsize=14, fontweight='bold')
    ax.set_title(f'COMSOL数据：粒子数密度随时间的演化 (共{num_timesteps}个时刻)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 根据时间点数量决定图例布局
    if num_timesteps <= 12:
        ncol = 2
    elif num_timesteps <= 24:
        ncol = 3
    else:
        ncol = 4
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=ncol)
    
    # 自动调整y轴范围
    all_densities_array = np.array(all_densities)
    y_min = np.min(all_densities_array)
    y_max = np.max(all_densities_array)
    y_range = y_max - y_min
    y_margin = max(y_range * 0.1, 1.0)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = 'validation'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'comsol_density_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*80}")
    print(f"图1（密度演化）已保存到: {output_path}")
    print(f"{'='*80}\n")
    
    # 显示图像（可选）
    # plt.show()
    plt.close()
    
    # ==================== 生成第二张图：相对密度变化 ====================
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # 计算相对于初始时刻的密度变化
    initial_density = all_densities[0]
    
    print(f"{'='*80}")
    print(f"开始计算相对密度变化（相对于初始时刻t={time_values[0]:.6f}s）...")
    print(f"{'='*80}\n")
    
    # 对所有时间点计算并绘制相对变化（包括初始时刻，变化为0%）
    for i, t in enumerate(time_values):
        density = all_densities[i]
        
        # 计算相对变化（百分比）
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (density - initial_density) / (initial_density + 1e-10) * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 绘制相对变化曲线
        label = f't = {t:.4f}s ({t*1000:.2f}ms)'
        ax2.plot(x_grid_mm, relative_change,
                color=colors[i],
                linewidth=2.0,
                alpha=0.8,
                label=label)
        
        print(f"时刻 {i+1}/{num_timesteps}: t={t:.6f}s, "
              f"相对变化范围=[{np.min(relative_change):.2f}%, {np.max(relative_change):.2f}%]")
    
    # 设置图表属性
    ax2.set_xlim(0, 34)
    ax2.set_xlabel('位置 x (mm)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('相对密度变化 (%)', fontsize=14, fontweight='bold')
    ax2.set_title(f'COMSOL数据：相对于初始时刻的粒子密度变化 (共{num_timesteps}条曲线)', 
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 根据曲线数量决定图例布局
    if num_timesteps <= 12:
        ncol2 = 2
    elif num_timesteps <= 24:
        ncol2 = 3
    else:
        ncol2 = 4
    
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, ncol=ncol2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # 保存第二张图
    output_path2 = os.path.join(output_dir, 'comsol_density_change.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    
    print(f"\n图2（相对密度变化）已保存到: {output_path2}")
    print(f"\n{'='*80}")
    print("✓ 可视化完成！")
    print(f"  - 共处理 {num_timesteps} 个时间点")
    print(f"  - 共处理 {num_particles} 个粒子")
    print(f"  - 图1: 绝对密度演化 ({num_timesteps}条曲线)")
    print(f"  - 图2: 相对密度变化 ({num_timesteps}条曲线)")
    print(f"{'='*80}\n")
    
    plt.close()


if __name__ == "__main__":
    main()
