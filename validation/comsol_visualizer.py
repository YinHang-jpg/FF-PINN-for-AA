import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os

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
        
        # 密度计算相关
        self.x_grid = None
        self.baseline_density = None
        self.bandwidth = 1.5  # 兼容旧方法；默认使用kNN估计
        
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
        """设置可视化界面"""
        # 创建图形和子图
        self.fig, (self.ax_particles, self.ax_density) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 计算x轴范围（基于实际粒子位置范围，以0为中心） - 使用NaN安全运算
        x_min_mm = np.nanmin(self.particle_positions) * 1000
        x_max_mm = np.nanmax(self.particle_positions) * 1000
        x_range = max(abs(x_min_mm), abs(x_max_mm))  # 取绝对值较大的作为范围
        x_center = (x_min_mm + x_max_mm) / 2  # 计算中心点
        if not np.isfinite(x_range):
            # 回退：如果仍非有限数，使用±17mm范围
            x_range = 17.0
            x_center = 0.0
        
        # 左侧子图：粒子位置（显示完整范围）
        self.ax_particles.set_xlim(x_center - x_range * 1.1, x_center + x_range * 1.1)  # 添加10%边距
        self.ax_particles.set_ylim(0, self.domain_size[1] * 1000)
        self.ax_particles.set_title("Particle Positions (COMSOL Data)")
        self.ax_particles.set_xlabel("x (mm)")
        self.ax_particles.set_ylabel("y (mm)")
        self.ax_particles.grid(True, alpha=0.3)
        
        # 右侧子图：密度分布（缩小到-17mm到+17mm）
        self.ax_density.set_xlim(-17, 17)
        # 初始y轴范围设置得更大，后续会动态调整
        self.ax_density.set_ylim(-1, 1)  # 初始相对变化百分比范围，后续动态调整
        self.ax_density.set_title("Particle Density Distribution")
        self.ax_density.set_xlabel("x (mm)")
        self.ax_density.set_ylabel("Relative Change (%)")
        self.ax_density.grid(True, alpha=0.3)
        
        # 创建x坐标网格用于密度计算（-17mm到+17mm），与DEM_main.py保持一致
        self.x_grid = np.linspace(-17, 17, 200)  # 与DEM_main.py中的网格密度一致
        
        # 计算初始密度分布作为基线（忽略NaN）
        initial_positions_mm = (self.particle_positions[0, :] * 1000)
        initial_positions_mm = initial_positions_mm[np.isfinite(initial_positions_mm)]
        initial_density = self.compute_density_knn(initial_positions_mm, self.x_grid, k_neighbors=None)
        self.baseline_density = np.mean(initial_density)
        
        # 初始化散点图和密度线
        self.scatter = self.ax_particles.scatter([], [], s=50, c='blue', alpha=0.6, label='Particles')
        self.density_line, = self.ax_density.plot([], [], 'b-', linewidth=3, label='Relative Change (%)', alpha=0.8)
        
        # 添加图例
        self.ax_particles.legend()
        self.ax_density.legend()
        
        # 添加时间显示
        self.time_text = self.ax_particles.text(0.02, 0.98, '', transform=self.ax_particles.transAxes, 
                                               verticalalignment='top', fontsize=12,
                                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def compute_density_knn(self, x_positions_mm, x_grid_mm, k_neighbors=None):
        """一维kNN密度估计，保持峰值并提供全局平滑。"""
        x_positions_mm = np.asarray(x_positions_mm)
        x_grid_mm = np.asarray(x_grid_mm)
        N = len(x_positions_mm)
        if N == 0:
            return np.zeros_like(x_grid_mm)
        if k_neighbors is None:
            k = max(10, int(0.03 * N))
        else:
            k = int(k_neighbors)
        k = max(1, min(k, max(1, N // 2)))
        diffs = np.abs(x_positions_mm[:, None] - x_grid_mm[None, :])
        r_k = np.partition(diffs, kth=k-1, axis=0)[k-1, :]
        density = k / (N * 2.0 * np.maximum(r_k, 1e-9))
        return density

    def smooth_preserve_minmax(self, y, window_ratio=0.07, min_limit=-99.9):
        """Hanning平滑并线性重标定到原始极值，且裁切到接近-100%。"""
        y = np.asarray(y)
        n = y.size
        if n < 5:
            return y
        w = max(5, int(n * float(window_ratio)))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        if w < 5:
            return y
        pad = w // 2
        y_pad = np.pad(y, (pad, pad), mode='reflect')
        kernel = np.hanning(w)
        kernel = kernel / np.sum(kernel)
        y_s = np.convolve(y_pad, kernel, mode='valid')
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        ys_min = float(np.nanmin(y_s))
        ys_max = float(np.nanmax(y_s))
        if np.isfinite(y_min) and np.isfinite(y_max) and ys_max > ys_min and y_max > y_min:
            a = (y_max - y_min) / (ys_max - ys_min)
            b = y_min - a * ys_min
            y_s = a * y_s + b
        y_s = np.maximum(y_s, min_limit)
        return y_s
    
    def update_frame(self, frame):
        """更新动画帧"""
        if frame >= self.num_frames:
            return self.scatter, self.density_line, self.time_text
        
        # 获取当前帧的粒子位置（忽略NaN）
        current_positions = (self.particle_positions[frame, :] * 1000)
        current_positions = current_positions[np.isfinite(current_positions)]
        
        # 更新散点图
        # 为每个粒子分配y坐标（均匀分布）
        y_positions = np.linspace(0, self.domain_size[1] * 1000, len(current_positions))
        scatter_data = np.column_stack([current_positions, y_positions])
        self.scatter.set_offsets(scatter_data)
        
        # 计算密度分布（kNN估计）并构造相对变化百分比
        current_density = self.compute_density_knn(current_positions, self.x_grid, k_neighbors=None)
        relative_change = (current_density - self.baseline_density) / self.baseline_density * 100.0
        relative_change = self.smooth_preserve_minmax(relative_change)
        
        # 更新密度曲线
        self.density_line.set_data(self.x_grid, relative_change)
        
        # 动态调整密度图的y轴范围（保持微小变化可见性）
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            
            # 如果数据范围很小，使用更精细的调整以显示微小变化
            if abs(y_max - y_min) < 1e-6:  # 如果范围小于1e-6
                # 对于非常小的变化，使用固定的精细范围
                center = (y_max + y_min) / 2.0
                y_margin = max(abs(center) * 0.1, 1e-7)  # 至少1e-7的边距
                self.ax_density.set_ylim(center - y_margin, center + y_margin)
            else:
                # 对于正常范围，使用10%边距
                y_margin = (y_max - y_min) * 0.1
                self.ax_density.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            # 如果没有有效数据，保持当前范围
            pass
        
        # 更新标题和时间显示
        current_time = self.times[frame]
        # 获取当前y轴范围用于显示
        y_lim = self.ax_density.get_ylim()
        y_range = y_lim[1] - y_lim[0]
        self.ax_density.set_title(f"Particle Density Distribution (t = {current_time:.6f} s)")
        self.time_text.set_text(f'Time: {current_time:.6f} s\nFrame: {frame+1}/{self.num_frames}\nParticles: {self.num_particles}\nY Range: [{y_lim[0]:.2e}, {y_lim[1]:.2e}]\nSpan: {y_range:.2e}')
        
        # 如果是最后一帧，输出密度相对变化的峰值与谷值及对应位置
        if frame == self.num_frames - 1:
            if np.any(np.isfinite(relative_change)):
                # 全局峰值（最大）与谷值（最小）
                peak_idx = int(np.nanargmax(relative_change))
                valley_idx = int(np.nanargmin(relative_change))
                peak_value = float(relative_change[peak_idx])
                valley_value = float(relative_change[valley_idx])
                peak_x = float(self.x_grid[peak_idx])
                valley_x = float(self.x_grid[valley_idx])
                print("\n=== 最后一帧密度相对变化峰谷信息 ===")
                print(f"峰值: {peak_value:.3f}% @ x = {peak_x:.3f} mm")
                print(f"谷值: {valley_value:.3f}% @ x = {valley_x:.3f} mm")
            else:
                print("\n=== 最后一帧密度相对变化峰谷信息 ===")
                print("数据均为NaN，无法计算峰谷")

        return self.scatter, self.density_line, self.time_text
    
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
        print("按Ctrl+C停止动画")
        ani = visualizer.show_animation(interval=100)  # 100ms间隔
        
    except KeyboardInterrupt:
        print("\n动画已停止")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
