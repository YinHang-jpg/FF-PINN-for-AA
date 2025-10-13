import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
from scipy.interpolate import PchipInterpolator

class COMSOLParticleVisualizer:
    def __init__(self, csv_file_path):
        """
        初始化COMSOL粒子可视化器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.times = None
        self.particle_positions = None
        self.domain_size = None
        self.num_particles = None
        self.num_frames = None
        
        # 动画相关
        self.fig = None
        self.ax_particles = None
        self.ax_density = None
        self.scatter = None
        self.density_line = None
        self.time_text = None
        
        # 图片保存相关
        self.save_images = False
        self.image_save_dir = "density_plots"
        self.last_saved_time = -1.0
        self.time_interval = 0.01  # 每0.01s保存一次图片
        
        # 密度计算相关
        self.x_grid = None
        self.baseline_density = None
        self.bandwidth = 1.5  # 保留字段但不再使用；改为KDE+优化带宽因子
        
        self.load_data()
        self.setup_visualization()
    
    def load_data(self):
        """加载CSV数据并进行NaN清洗"""
        try:
            # 读取CSV文件（允许空白行），先按字符串读入再统一转换为数值，无法转换的置为NaN
            raw = pd.read_csv(self.csv_file_path, header=None, skip_blank_lines=True)
            raw = raw.dropna(axis=1, how='all')  # 丢弃全为空的列（常见于尾部多余逗号）
            raw = raw.apply(pd.to_numeric, errors='coerce')  # 全部转为数值，非法值->NaN

            # 丢弃时间列为NaN的行
            raw = raw.dropna(axis=0, subset=[0])

            # 提取时间和粒子位置
            self.times = raw.iloc[:, 0].to_numpy()
            positions_df = raw.iloc[:, 1:]

            # 丢弃全为NaN的粒子列（某些列可能完全为空）
            positions_df = positions_df.dropna(axis=1, how='all')

            self.particle_positions = positions_df.to_numpy()  # 形状: [帧数, 粒子数]，允许含NaN

            # 获取数据维度
            self.num_frames = len(self.times)
            self.num_particles = self.particle_positions.shape[1]

            # 使用NaN安全的min/max进行范围打印
            try:
                pos_min = float(np.nanmin(self.particle_positions))
                pos_max = float(np.nanmax(self.particle_positions))
                print(f"数据加载成功:")
                print(f"  时间帧数: {self.num_frames}")
                print(f"  粒子数量: {self.num_particles}")
                print(f"  时间范围: {self.times[0]:.6f}s - {self.times[-1]:.6f}s")
                print(f"  位置范围: {pos_min*1000:.3f}mm - {pos_max*1000:.3f}mm")
            except ValueError:
                # 全部为NaN的极端情况
                pos_min, pos_max = -0.017, 0.017
                print(f"数据加载成功，但位置数据均为NaN，使用默认范围±17mm")

            # 计算域大小（基于粒子位置范围）
            x_min = pos_min
            x_max = pos_max
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                x_min, x_max = -0.017, 0.017  # 回退到±17mm（米）
            y_range = 0.01  # 假设y方向范围（米）
            self.domain_size = (float(x_max - x_min + 0.002), y_range)  # 添加一些边距

        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def setup_visualization(self):
        """设置可视化界面 - 使用DEM_main.py的绘图机制"""
        # 创建图形和子图
        self.fig, (self.ax_particles, self.ax_density) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左侧子图：粒子运动
        self.ax_particles.set_xlim(0.0, 34.0)
        self.ax_particles.set_ylim(0, self.domain_size[1] * 1000)
        self.ax_particles.set_title("Particle Motion (COMSOL Data)")
        self.ax_particles.set_xlabel("x (mm)")
        self.ax_particles.set_ylabel("y (mm)")
        
        # 右侧子图：密度变化曲线
        self.ax_density.set_xlim(0.0, 34.0)
        self.ax_density.set_ylim(-5, 5)  # 进一步缩小y轴范围，确保曲线可见
        self.ax_density.set_title("Particle Density Distribution")
        self.ax_density.set_xlabel("x (mm)")
        self.ax_density.set_ylabel("Relative Change (%)")
        self.ax_density.grid(True, alpha=0.3)
        
        # 初始化基于距离的浓度统计机制
        # 三个参考位置：x=0mm, x=7.5mm, x=15mm
        self.reference_positions = np.array([0.0, 8.5, 25.5])  # mm
        self.num_references = len(self.reference_positions)
        
        # 浓度统计参数
        self.concentration_scale = 100.0  # 浓度缩放因子
        self.distance_weight = 1.0  # 距离权重
        self.baseline_concentration = 0.0  # 基准浓度（相对变化）
        
        # 创建x轴网格用于显示浓度分布
        self.x_grid = np.linspace(0.0, 34.0, 100)  # 100个点用于平滑显示
        self.concentration_values = np.zeros_like(self.x_grid)
        
        # 计算初始浓度分布
        self.initial_concentrations = self._calculate_concentration_distribution(0)
        self.baseline_concentration = np.mean(self.initial_concentrations)
        
        # 调试信息：打印浓度统计设置
        print(f"\n[调试] 基于距离的浓度统计设置:")
        print(f"  参考位置: {self.reference_positions} mm")
        print(f"  浓度缩放因子: {self.concentration_scale}")
        print(f"  距离权重: {self.distance_weight}")
        print(f"  基准浓度: {self.baseline_concentration:.3f}")
        print(f"  总粒子数: {self.num_particles}")
        
        # 初始化散点图
        self.scatter = self.ax_particles.scatter([], [], s=50, c='blue', alpha=0.6, label='Particles')
        
        # 初始化基于距离的浓度柱状图
        # 初始化柱状图（使用x_grid作为显示）
        self.bars = self.ax_density.bar(self.x_grid, [0]*len(self.x_grid), width=0.34, align='center', alpha=0.8, label='Concentration Change')
        self.curve_line, = self.ax_density.plot([], [], 'r-', linewidth=2, label='Smoothed')
        
        # 添加图例
        self.ax_particles.legend()
        self.ax_density.legend()
        
        # 添加时间显示
        self.time_text = self.ax_particles.text(0.02, 0.98, '', transform=self.ax_particles.transAxes, 
                                               verticalalignment='top', fontsize=10,
                                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 打印区间内粒子数量在 t=0 与结束时的对比
        self.print_initial_final_bin_counts()
        
        # 设置图片保存功能
        self.setup_image_saving()
    
    def _calculate_concentration_distribution(self, frame):
        """基于粒子到参考位置的距离计算浓度分布"""
        try:
            # 获取当前帧的粒子位置（单位：mm）
            current_positions = self.particle_positions[frame, :] * 1000.0
            current_positions = current_positions[np.isfinite(current_positions)]
            
            if len(current_positions) == 0:
                return np.zeros_like(self.x_grid)
            
            # 计算每个x_grid位置处的浓度
            concentrations = np.zeros_like(self.x_grid)
            
            for i, x_pos in enumerate(self.x_grid):
                # 计算所有粒子到当前x位置的距离
                distances = np.abs(current_positions - x_pos)
                
                # 浓度计算：距离越近，浓度越高
                # 使用高斯函数形式的浓度分布
                sigma = 0.2  # 浓度分布的标准差（mm）
                concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
                
                concentrations[i] = concentration
            
            return concentrations
        except Exception as e:
            print(f"浓度计算错误 (frame {frame}): {e}")
            return np.zeros_like(self.x_grid)
    
    def print_initial_final_bin_counts(self):
        """打印基于距离的浓度统计信息"""
        try:
            # 计算初始和最终浓度分布
            initial_concentrations = self._calculate_concentration_distribution(0)
            final_concentrations = self._calculate_concentration_distribution(self.num_frames - 1)
            
            # 计算相对变化
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_change = (final_concentrations - initial_concentrations) / initial_concentrations * 100.0
                relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 打印表头
            print("\n[验证] 基于距离的浓度统计：t=0 vs 结束时")
            print(f"  参考位置: {self.reference_positions} mm")
            print(f"  粒子总数: {self.num_particles}")
            print("  位置(mm)                初始浓度    最终浓度    相对变化(%)")

            # 选择关键位置进行显示
            key_positions = [0.0, 8.5, 17.0, 25.5, 34.0]
            for pos in key_positions:
                # 找到最接近的x_grid索引
                idx = np.argmin(np.abs(self.x_grid - pos))
                c0 = initial_concentrations[idx]
                cT = final_concentrations[idx]
                change = relative_change[idx]
                print(f"  {pos:6.1f}                    {c0:8.3f}    {cT:8.3f}    {change:+8.1f}")

            # 计算整体统计
            avg_initial = np.mean(initial_concentrations)
            avg_final = np.mean(final_concentrations)
            avg_change = np.mean(relative_change)
            
            print("  ----------------------------------------------")
            print(f"  平均浓度                 {avg_initial:8.3f}    {avg_final:8.3f}    {avg_change:+8.1f}")
            
            # 打印峰值位置的变化
            print(f"\n[峰值分析] 参考位置附近的浓度变化:")
            for i, ref_pos in enumerate(self.reference_positions):
                idx = np.argmin(np.abs(self.x_grid - ref_pos))
                c0 = initial_concentrations[idx]
                cT = final_concentrations[idx]
                change = relative_change[idx]
                print(f"  参考位置{i+1} ({ref_pos:5.1f}mm): 初始={c0:6.3f} 最终={cT:6.3f} 变化={change:+6.1f}%")
            print()
        except Exception as e:
            print(f"[验证打印失败]: {e}")

    def setup_image_saving(self):
        """设置图片保存功能"""
        import os
        import shutil
        
        # 创建保存目录
        if os.path.exists(self.image_save_dir):
            # 如果目录存在，清空所有文件
            shutil.rmtree(self.image_save_dir)
            print(f"清空现有图片目录: {self.image_save_dir}")
        
        os.makedirs(self.image_save_dir, exist_ok=True)
        print(f"创建图片保存目录: {self.image_save_dir}")
        
        # 启用图片保存
        self.save_images = True
        self.last_saved_time = -1.0
        
    def save_density_image(self, frame, current_time):
        """保存当前密度曲线图片 - 直接保存动画帧"""
        import os
        
        if not self.save_images:
            return
            
        # 检查是否到了保存时间间隔（基于精确的0.01s倍数）
        target_time = round(current_time / self.time_interval) * self.time_interval
        
        # 只有当当前时间接近目标时间且距离上次保存足够远时才保存
        if (abs(current_time - target_time) > 0.001 or 
            current_time - self.last_saved_time < self.time_interval - 0.001):
            return
            
        try:
            # 确保目录存在
            os.makedirs(self.image_save_dir, exist_ok=True)
            
            # 直接保存当前动画帧
            filename = f"density_plot_{frame:03d}_t_{current_time:.6f}s.png"
            filepath = os.path.join(self.image_save_dir, filename)
            
            # 保存整个图形（包含粒子运动和密度两个子图）
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            
            # 更新最后保存时间
            self.last_saved_time = current_time
            
            print(f"保存密度图片: {filename}")
            print(f"  保存路径: {os.path.abspath(filepath)}")
            
        except Exception as e:
            print(f"保存图片失败 (frame {frame}): {e}")

    
    def update_frame(self, frame):
        """更新动画帧 - 使用DEM_main.py的绘图机制"""
        if frame >= self.num_frames:
            return self.scatter, self.curve_line, self.time_text
        
        # 获取当前帧的粒子位置（忽略NaN）
        current_positions = (self.particle_positions[frame, :] * 1000)
        current_positions = current_positions[np.isfinite(current_positions)]
        
        # 更新散点图
        # 为每个粒子分配y坐标（均匀分布）
        y_positions = np.linspace(0, self.domain_size[1] * 1000, len(current_positions))
        scatter_data = np.column_stack([current_positions, y_positions])
        self.scatter.set_offsets(scatter_data)
        
        # 更新基于距离的浓度分布
        current_concentrations = self._calculate_concentration_distribution(frame)
        
        # 计算相对变化
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (current_concentrations - self.initial_concentrations) / self.initial_concentrations * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新每个bar的高度
        for rect, h in zip(self.bars, relative_change):
            rect.set_height(h)

        # 更新平滑曲线（PCHIP 连接柱顶）
        try:
            pchip = PchipInterpolator(self.x_grid, relative_change, extrapolate=False)
            x_smooth = np.linspace(self.x_grid[0], self.x_grid[-1], max(50, 4 * len(self.x_grid)))
            y_smooth = pchip(x_smooth)
            self.curve_line.set_data(x_smooth, y_smooth)
        except Exception:
            self.curve_line.set_data(self.x_grid, relative_change)

        # 动态调整y轴范围
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            self.ax_density.set_ylim(y_min - y_margin, y_max + y_margin)

        # 更新标题和时间显示
        current_time = self.times[frame]
        self.ax_density.set_title(f"Particle Density Change Rate (t = {current_time:.6f} s)")
        self.time_text.set_text(f'Time: {current_time:.6f} s\nFrame: {frame+1}/{self.num_frames}\nParticles: {len(current_positions)}')

        # 保存密度图片（每0.01s保存一次）
        self.save_density_image(frame, current_time)

        return self.scatter, self.curve_line, self.time_text
    
    def create_animation(self, interval=50, save_path=None):
        """
        创建动画
        
        Args:
            interval: 动画间隔（毫秒）
            save_path: 保存路径（可选）
        """
        print(f"创建动画...")
        print(f"  总帧数: {self.num_frames}")
        print(f"  动画间隔: {interval}ms")
        print(f"  预计时长: {self.num_frames * interval / 1000:.1f}秒")
        
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=self.num_frames,
            interval=interval,
            blit=False,
            repeat=False
        )
        
        # 保存动画（如果指定了路径）
        if save_path:
            print(f"保存动画到: {save_path}")
            ani.save(save_path, writer='pillow', fps=1000//interval)
            print("动画保存完成")
        
        return ani
    
    def show_animation(self, interval=50):
        """显示动画"""
        ani = self.create_animation(interval=interval)
        plt.tight_layout()
        plt.show()
        return ani

def main():
    """主函数"""
    # CSV文件路径
    csv_file = "validation/comsol_positions.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_file):
        print(f"错误: 文件 {csv_file} 不存在")
        return
    
    try:
        # 创建可视化器
        print("创建COMSOL粒子可视化器...")
        visualizer = COMSOLParticleVisualizer(csv_file)
        
        # 显示动画
        print("\n开始播放动画...")
        print("密度曲线图片将每0.01s自动保存到 'density_plots' 目录")
        print("按Ctrl+C停止动画")
        ani = visualizer.show_animation(interval=100)  # 100ms间隔
        
    except KeyboardInterrupt:
        print("\n动画已停止")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
