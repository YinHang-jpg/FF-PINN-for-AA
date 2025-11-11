# particle_initialization.py
import numpy as np
import math


def _min_distance_random_sampling(width, height, min_dist, target_count, rng):
    """
    生成均匀随机分布的采样点，同时保证任意两点之间的距离不少于 min_dist。
    采用格点加邻域检测的拒绝采样方式，可覆盖整个区域。

    :param width: 区域宽度 (x 方向长度)
    :param height: 区域高度 (y 方向长度)
    :param min_dist: 最小允许距离
    :param target_count: 需要生成的点数
    :param rng: NumPy 随机生成器
    :return: (target_count, 2) 的数组，每行是一个点的 (x, y)
    """
    if target_count == 0:
        return np.zeros((0, 2), dtype=float)

    cell_size = min_dist
    grid_w = max(1, int(np.ceil(width / cell_size)))
    grid_h = max(1, int(np.ceil(height / cell_size)))
    grid = -np.ones((grid_w, grid_h), dtype=int)

    samples = []
    min_dist_sq = min_dist * min_dist

    def _grid_coords(point):
        gx = int(point[0] / cell_size)
        gy = int(point[1] / cell_size)
        return min(gx, grid_w - 1), min(gy, grid_h - 1)

    max_total_attempts = max(10 * target_count, 1000)
    attempts = 0

    while len(samples) < target_count and attempts < max_total_attempts:
        candidate = rng.uniform([0.0, 0.0], [width, height])
        attempts += 1

        cgx, cgy = _grid_coords(candidate)
        x_min = max(cgx - 1, 0)
        x_max = min(cgx + 2, grid_w)
        y_min = max(cgy - 1, 0)
        y_max = min(cgy + 2, grid_h)

        too_close = False
        for gx in range(x_min, x_max):
            for gy in range(y_min, y_max):
                neighbor_idx = grid[gx, gy]
                if neighbor_idx == -1:
                    continue
                neighbor_point = samples[neighbor_idx]
                dx = candidate[0] - neighbor_point[0]
                dy = candidate[1] - neighbor_point[1]
                if dx * dx + dy * dy < min_dist_sq:
                    too_close = True
                    break
            if too_close:
                break

        if too_close:
            continue

        samples.append(candidate)
        grid[cgx, cgy] = len(samples) - 1

    if len(samples) < target_count:
        raise RuntimeError(
            f"无法在给定的最小距离 {min_dist} 内生成 {target_count} 个不重叠粒子，请减少粒子数量或放大域尺寸。"
        )

    return np.asarray(samples, dtype=float)


def initialize_particles(N=0, domain_size=(0.01, 0.01), diameter=2e-6, density=2000, init_mode='linear'):
    """
    初始化粒子群
    
    :param N: 粒子数量
    :param domain_size: 模拟域尺寸 (Lx, Ly)，单位：米
    :param diameter: 每个粒子的直径，单位：米
    :param density: 每个粒子的密度，单位：kg/m^3
    :param init_mode: 初始化模式
        - 'linear': 直线排布（在域内均匀分布）
        - 'random': 在计算域内随机分布
        - 'uniform': 在计算域内均匀分布
        - 'sample': 样例分布（所有粒子竖着堆砌在 x=8.5mm 与 x=25.5mm）
        - 'normal': 正态分布（仅沿 x 轴，以域中心为均值；y=0）
    :return: 位置 (N,2)、速度 (N,2)、半径数组 (N,), 质量数组 (N,)
    """
    Lx, Ly = domain_size
    
    if init_mode == 'linear':
        # 直线排布模式：在域内均匀分布粒子
        # 在域内均匀分布粒子，不扩展到域外
        x_positions = np.linspace(0, Lx, N, endpoint=True)
        y_position = Ly / 2  # y坐标固定在域中心
        positions = np.column_stack((x_positions, np.full(N, y_position)))
        
    elif init_mode == 'random':
        # 随机分布模式：使用拒绝采样 + 网格加速，保证粒子之间至少相距5微米
        min_center_distance = 5e-6  # 5 微米
        rng = np.random.default_rng()
        positions = _min_distance_random_sampling(Lx, Ly, min_center_distance, N, rng)
        
    elif init_mode == 'uniform':
        # 均匀分布模式：在计算域内均匀分布粒子
        # 计算网格尺寸，使粒子尽可能均匀分布
        grid_size = int(np.ceil(np.sqrt(N)))
        x_grid = np.linspace(0, Lx, grid_size, endpoint=False)
        y_grid = np.linspace(0, Ly, grid_size, endpoint=False)
        
        # 创建网格点
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        # 如果网格点数量超过N，随机选择N个点
        if len(grid_points) > N:
            indices = np.random.choice(len(grid_points), N, replace=False)
            positions = grid_points[indices]
        else:
            # 如果网格点数量不足N，随机添加额外点
            positions = grid_points.copy()
            additional_N = N - len(positions)
            additional_x = np.random.uniform(0, Lx, additional_N)
            additional_y = np.random.uniform(0, Ly, additional_N)
            additional_positions = np.column_stack((additional_x, additional_y))
            positions = np.vstack((positions, additional_positions))
    elif init_mode == 'sample':
        # 样例分布：将粒子竖直堆砌在两条固定的x位置：8.5mm与25.5mm
        x_left = 0.0085   # 8.5 mm
        x_right = 0.0255  # 25.5 mm
        n_left = N // 2
        n_right = N - n_left

        # 在y方向上均匀分布（不包含上边界，避免重合）
        if n_left > 0:
            y_left = np.linspace(0, Ly, n_left, endpoint=False)
            left_positions = np.column_stack((np.full(n_left, x_left), y_left))
        else:
            left_positions = np.empty((0, 2))

        if n_right > 0:
            y_right = np.linspace(0, Ly, n_right, endpoint=False)
            right_positions = np.column_stack((np.full(n_right, x_right), y_right))
        else:
            right_positions = np.empty((0, 2))

        positions = np.vstack((left_positions, right_positions)) if N > 0 else np.zeros((0, 2))
    elif init_mode == 'normal':
        # 正态分布（仅在x轴上，截断到域内）且确定性：使用截断正态分位数均匀取点
        # 均值取域中心，标准差取域长的1/10；严格位于[0, Lx)且无随机性
        mu_x = Lx / 2.0
        sigma_x = Lx / 10.0

        # 标准正态CDF与其近似逆（Acklam算法），避免额外依赖
        def _phi(x):
            x = np.asarray(x, dtype=float)
            erf_vec = np.vectorize(math.erf)
            return 0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))

        def _phi_inv(p):
            # 源自 Peter J. Acklam 的近似，适用于 0<p<1
            a = np.array([-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
                          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00])
            b = np.array([-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
                          6.680131188771972e+01, -1.328068155288572e+01])
            c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00])
            d = np.array([ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                           3.754408661907416e+00])

            plow = 0.02425
            phigh = 1 - plow
            p = np.asarray(p)
            x = np.empty_like(p, dtype=float)

            # 低端
            mask_low = p < plow
            if np.any(mask_low):
                q = np.sqrt(-2.0 * np.log(p[mask_low]))
                x[mask_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

            # 中间
            mask_mid = (p >= plow) & (p <= phigh)
            if np.any(mask_mid):
                q = p[mask_mid] - 0.5
                r = q*q
                x[mask_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

            # 高端
            mask_high = p > phigh
            if np.any(mask_high):
                q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
                x[mask_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

            return x

        # 截断到[0, Lx)的分位区间
        a = (0.0 - mu_x) / sigma_x
        b = (Lx - mu_x) / sigma_x
        Fa = _phi(a)
        Fb = _phi(b)

        # 在截断CDF范围内均匀取分位点，使用(i+0.5)/N 避免端点
        p = (np.arange(N, dtype=float) + 0.5) / max(N, 1)
        u = Fa + p * (Fb - Fa)
        z = _phi_inv(u)
        x_positions = mu_x + sigma_x * z
        # 理论上已在域内，数值上做一次极小裁剪
        eps = np.finfo(float).eps
        x_positions = np.clip(x_positions, 0.0, max(0.0, Lx - eps))

        # y位置保持为当前固定值（用户设定为0.01）
        y_positions = np.ones(N) * 0.01
        positions = np.column_stack((x_positions, y_positions))
    else:
        raise ValueError(f"不支持的初始化模式: {init_mode}。支持的模式: 'linear', 'random', 'uniform', 'sample', 'normal'")

    velocities = np.zeros_like(positions)
    radii = np.ones(N) * (diameter / 2)
    # 计算质量（三维近似球体：m = ρ * 4/3 * π * r^3）
    mass = density * 4/3 * np.pi * radii**3
    
    return positions, velocities, radii, mass


def test_initialization_modes():
    """
    测试不同初始化模式的效果
    """
    import matplotlib.pyplot as plt
    
    # 测试参数
    N = 100
    domain_size = (0.01, 0.01)  # 1cm x 1cm
    diameter = 2e-6
    density = 2000
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['linear', 'random', 'uniform']
    mode_names = ['直线排布', '随机分布', '均匀分布']
    
    for i, (mode, name) in enumerate(zip(modes, mode_names)):
        positions, velocities, radii, mass = initialize_particles(
            N=N, domain_size=domain_size, diameter=diameter, 
            density=density, init_mode=mode
        )
        
        # 绘制粒子位置
        axes[i].scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, 
                       s=20, alpha=0.7, c='blue')
        axes[i].set_xlim(0, domain_size[0] * 1000)
        axes[i].set_ylim(0, domain_size[1] * 1000)
        axes[i].set_xlabel('x (mm)')
        axes[i].set_ylabel('y (mm)')
        axes[i].set_title(f'{name} (N={N})')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("初始化模式测试结果:")
    print(f"域大小: {domain_size[0]*1000:.1f}mm x {domain_size[1]*1000:.1f}mm")
    print(f"粒子数量: {N}")
    print(f"粒子直径: {diameter*1e6:.1f}μm")
    print(f"粒子密度: {density} kg/m³")


if __name__ == "__main__":
    test_initialization_modes()