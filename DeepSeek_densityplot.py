import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar

# Configure matplotlib to support Chinese characters and proper minus sign rendering
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

def compute_dynamic_ylim(values, padding_ratio=0.1):
    """
    根据数据范围动态计算y轴上下限，并留出一定边距，确保包含0。
    """
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    span = vmax - vmin
    pad = max(span * padding_ratio, 1.0)
    y_min = min(vmin - pad, 0.0)
    y_max = max(vmax + pad, 0.0)
    return y_min, y_max

def compute_concentration_curve(positions, bandwidth_method='optimized', smooth_curve=True):
    """
    计算粒子浓度的空间分布曲线
    
    参数:
    positions: 粒子位置数组，形状为(N, 2)
    bandwidth_method: 带宽选择方法，'silverman', 'scott', 或 'optimized'
    smooth_curve: 是否对曲线进行后处理平滑
    
    返回:
    x_vals: 空间坐标数组
    concentration: 浓度百分比偏差曲线
    """
    # 提取x坐标（假设我们关注x方向的浓度分布）
    x_coords = positions[:, 0]
    
    # 移除可能的NaN或无穷大值
    x_coords = x_coords[np.isfinite(x_coords)]
    
    if len(x_coords) < 2:
        raise ValueError("需要至少2个有效的粒子位置")
    
    # 定义评估点范围（扩展到数据范围之外一点以获得边界效应）
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_range = x_max - x_min
    x_vals = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 1000)
    
    # 选择带宽
    if bandwidth_method == 'silverman':
        # Silverman规则
        bw = len(x_coords)**(-1/5) * np.std(x_coords)
    elif bandwidth_method == 'scott':
        # Scott规则
        bw = len(x_coords)**(-1/5) * 1.06 * np.std(x_coords)
    elif bandwidth_method == 'optimized':
        # 使用交叉验证优化带宽
        bw = optimize_bandwidth(x_coords)
    else:
        bw = bandwidth_method  # 直接使用给定的数值
    
    print(f"使用的带宽: {bw:.4f}")
    
    # 计算核密度估计
    kde = gaussian_kde(x_coords, bw_method=bw)
    density = kde(x_vals)
    
    # 归一化为百分比偏差: (局部密度 - 平均密度) / 平均密度 * 100%
    mean_density = np.mean(density)
    concentration = (density - mean_density) / mean_density * 100
    
    # 后处理平滑（可选）
    if smooth_curve and len(concentration) > 11:
        concentration = savgol_filter(concentration, window_length=min(11, len(concentration)-2), polyorder=3)
    
    return x_vals, concentration

def optimize_bandwidth(x_coords, n_folds=5):
    """
    使用交叉验证优化KDE带宽
    
    参数:
    x_coords: 粒子x坐标
    n_folds: 交叉验证折数
    
    返回:
    最优带宽
    """
    def objective(bw):
        """目标函数：负对数似然"""
        if bw <= 0:
            return np.inf
        
        # 使用留一法交叉验证
        n = len(x_coords)
        log_likelihood = 0
        
        for i in range(min(n, 20)):  # 为了效率，只使用部分数据点
            # 留出第i个点
            train_data = np.delete(x_coords, i)
            test_point = x_coords[i]
            
            # 在训练数据上计算KDE
            try:
                kde = gaussian_kde(train_data, bw_method=bw)
                # 计算测试点的对数似然
                pdf_val = kde(test_point)[0]
                if pdf_val > 0:
                    log_likelihood += np.log(pdf_val)
            except:
                pass
        
        return -log_likelihood  # 最小化负对数似然
    
    # 在合理范围内搜索最优带宽
    silverman_bw = len(x_coords)**(-1/5) * np.std(x_coords)
    result = minimize_scalar(objective, bounds=(silverman_bw/10, silverman_bw*10), method='bounded')
    
    return result.x

def generate_test_particles(Lx=1.0, Ly=1.0, N=1000):
    """
    生成测试用的粒子分布（模拟您图中可能的多峰分布）
    这用于演示，您应该用您自己的粒子数据替换
    """
    # 创建双峰分布（模拟图中的两个峰值）
    np.random.seed(42)  # 可重复性
    
    # 第一个高斯峰
    n1 = N // 2
    x1 = np.random.normal(Lx * 0.3, Lx * 0.05, n1)
    y1 = np.random.normal(Ly * 0.5, Ly * 0.1, n1)
    
    # 第二个高斯峰
    n2 = N - n1
    x2 = np.random.normal(Lx * 0.7, Lx * 0.05, n2)
    y2 = np.random.normal(Ly * 0.5, Ly * 0.1, n2)
    
    # 合并
    x_positions = np.concatenate([x1, x2])
    y_positions = np.concatenate([y1, y2])
    
    positions = np.column_stack((x_positions, y_positions))
    
    return positions

def plot_concentration_comparison(positions, title="浓度分布曲线"):
    """
    绘制不同带宽设置下的浓度曲线比较
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # 不同带宽设置
    bandwidth_settings = [
        ('silverman', 'Silverman规则'),
        ('scott', 'Scott规则'),
        ('optimized', '优化带宽'),
        (0.01, '小带宽(0.01)')
    ]
    
    for i, (bw_method, bw_name) in enumerate(bandwidth_settings):
        try:
            x_vals, concentration = compute_concentration_curve(
                positions, 
                bandwidth_method=bw_method,
                smooth_curve=True
            )
            
            axes[i].plot(x_vals, concentration, 'b-', linewidth=2, label=bw_name)
            axes[i].set_ylabel('浓度 (%)')
            axes[i].set_xlabel('空间位置')
            axes[i].set_title(f'{bw_name}\n峰值: {np.max(concentration):.1f}%, 谷值: {np.min(concentration):.1f}%')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 1)
            dyn_ylim = compute_dynamic_ylim(concentration)
            if dyn_ylim is not None:
                axes[i].set_ylim(*dyn_ylim)
            axes[i].legend()
        except Exception as e:
            axes[i].text(0.5, 0.5, f'错误: {str(e)}', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == "__main__":
    # 参数设置
    Lx, Ly = 1.0, 1.0  # 域大小
    N = 1000  # 粒子数
    
    # 方法1: 使用您提供的初始化代码（线性分布）
    # 注意：线性分布可能不会产生高峰值，这里仅作演示
    print("=== 方法1: 使用线性初始化 ===")
    sound_speed = 1.0
    frequency = 1.0
    init_mode = 'linear'
    
    if init_mode == 'linear':
        wavelength = sound_speed / frequency
        quarter_wavelength = wavelength / 4
        x_positions = np.linspace(-quarter_wavelength, Lx + quarter_wavelength, N, endpoint=True)
        y_position = Ly / 2
        linear_positions = np.column_stack((x_positions, np.full(N, y_position)))
    
    # 计算并绘制线性分布的浓度曲线
    try:
        x_vals_linear, conc_linear = compute_concentration_curve(linear_positions)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals_linear, conc_linear, 'b-', linewidth=2, label='线性分布')
        plt.ylabel('浓度 (%)')
        plt.xlabel('空间位置')
        plt.title('线性粒子分布的浓度曲线')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.legend()
        plt.show()
        
        print(f"线性分布 - 峰值: {np.max(conc_linear):.1f}%, 谷值: {np.min(conc_linear):.1f}%")
    except Exception as e:
        print(f"线性分布计算错误: {e}")
    
    # 方法2: 使用测试用的多峰分布（更接近您描述的情况）
    print("\n=== 方法2: 使用多峰测试分布 ===")
    test_positions = generate_test_particles(Lx, Ly, N)
    
    # 绘制粒子分布散点图
    plt.figure(figsize=(10, 4))
    plt.scatter(test_positions[:, 0], test_positions[:, 1], s=1, alpha=0.6)
    plt.xlabel('X位置')
    plt.ylabel('Y位置')
    plt.title('粒子分布散点图')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.show()
    
    # 计算并绘制浓度曲线
    x_vals, concentration = compute_concentration_curve(test_positions)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, concentration, 'r-', linewidth=2, label='浓度曲线')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='平均浓度')
    plt.ylabel('浓度 (%)')
    plt.xlabel('空间位置')
    plt.title('粒子浓度分布曲线')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    dyn_ylim = compute_dynamic_ylim(concentration)
    if dyn_ylim is not None:
        plt.ylim(*dyn_ylim)
    plt.legend()
    plt.show()
    
    print(f"测试分布 - 峰值: {np.max(concentration):.1f}%, 谷值: {np.min(concentration):.1f}%")
    
    # 方法3: 比较不同带宽设置
    print("\n=== 方法3: 不同带宽设置比较 ===")
    plot_concentration_comparison(test_positions)
    
    # 保存结果（可选）
    np.savetxt('concentration_curve.txt', np.column_stack((x_vals, concentration)), 
               header='x_position concentration_percent', fmt='%.6f')

# 使用说明:
# 1. 将您自己的粒子位置数据替换掉 generate_test_particles() 函数
# 2. 调整带宽方法: 'silverman', 'scott', 'optimized', 或直接指定数值
# 3. 如果曲线不够平滑，可以调整 savgol_filter 的参数