import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# 内置数据：您提供的粒子位置（两列：x, y）。第一列为 x。
EMBEDDED_POSITIONS = np.array([
    [-0.0085, 0.017],
    [-0.00798485, 0.017],
    [-0.00746969, 0.017],
    [-0.00695454, 0.017],
    [-0.00643938, 0.017],
    [-0.00592423, 0.017],
    [-0.00540907, 0.017],
    [-0.00489392, 0.017],
    [-0.00437876, 0.017],
    [-0.00386361, 0.017],
    [-0.00334846, 0.017],
    [-0.0028333, 0.017],
    [-0.00231815, 0.017],
    [-0.001803, 0.017],
    [-0.00128784, 0.017],
    [-0.00077269, 0.017],
    [-0.00025754, 0.017],
    [0.00025762, 0.017],
    [0.00077277, 0.017],
    [0.00128792, 0.017],
    [0.00180307, 0.017],
    [0.00231822, 0.017],
    [0.00283337, 0.017],
    [0.00334852, 0.017],
    [0.00386367, 0.017],
    [0.00437882, 0.017],
    [0.00489397, 0.017],
    [0.00540912, 0.017],
    [0.00592426, 0.017],
    [0.00643941, 0.017],
    [0.00695456, 0.017],
    [0.00746971, 0.017],
    [0.00798485, 0.017],
    [0.0085, 0.017],
    [0.00901515, 0.017],
    [0.00953029, 0.017],
    [0.01004544, 0.017],
    [0.01056059, 0.017],
    [0.01107574, 0.017],
    [0.01159088, 0.017],
    [0.01210603, 0.017],
    [0.01262118, 0.017],
    [0.01313633, 0.017],
    [0.01365148, 0.017],
    [0.01416663, 0.017],
    [0.01468178, 0.017],
    [0.01519693, 0.017],
    [0.01571208, 0.017],
    [0.01622723, 0.017],
    [0.01674238, 0.017],
    [0.01725754, 0.017],
    [0.01777269, 0.017],
    [0.01828784, 0.017],
    [0.018803, 0.017],
    [0.01931815, 0.017],
    [0.0198333, 0.017],
    [0.02034846, 0.017],
    [0.02086361, 0.017],
    [0.02137876, 0.017],
    [0.02189392, 0.017],
    [0.02240907, 0.017],
    [0.02292423, 0.017],
    [0.02343938, 0.017],
    [0.02395454, 0.017],
    [0.02446969, 0.017],
    [0.02498485, 0.017],
    [0.0255, 0.017],
    [0.02601515, 0.017],
    [0.02653031, 0.017],
    [0.02704546, 0.017],
    [0.02756062, 0.017],
    [0.02807577, 0.017],
    [0.02859093, 0.017],
    [0.02910608, 0.017],
    [0.02962124, 0.017],
    [0.03013639, 0.017],
    [0.03065154, 0.017],
    [0.0311667, 0.017],
    [0.03168185, 0.017],
    [0.032197, 0.017],
    [0.03271216, 0.017],
    [0.03322731, 0.017],
    [0.03374246, 0.017],
    [0.03425762, 0.017],
    [0.03477277, 0.017],
    [0.03528792, 0.017],
    [0.03580307, 0.017],
    [0.03631822, 0.017],
    [0.03683337, 0.017],
    [0.03734852, 0.017],
    [0.03786367, 0.017],
    [0.03837882, 0.017],
    [0.03889397, 0.017],
    [0.03940912, 0.017],
    [0.03992426, 0.017],
    [0.04043941, 0.017],
    [0.04095456, 0.017],
    [0.04146971, 0.017],
    [0.04198485, 0.017],
    [0.0425, 0.017],
])


def load_x_positions(file_path, column_index=0):
    """
    从文件加载粒子 x 位置。支持 .csv/.txt（分隔符自动推断）和 .npy。

    :param file_path: 文件路径
    :param column_index: 取第几列作为 x（默认第 0 列）
    :return: (N,) numpy 数组，单位与文件一致
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npy':
        data = np.load(file_path)
    else:
        # 尝试以常见分隔符读取
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except Exception:
            try:
                data = np.loadtxt(file_path)  # 空白分隔
            except Exception as e:
                raise RuntimeError(f"无法读取文件: {file_path} - {e}")

    data = np.atleast_2d(data)
    if data.shape[1] <= column_index:
        raise ValueError(f"数据列数不足，无法获取第 {column_index} 列。实际列数: {data.shape[1]}")
    x = data[:, column_index]
    return np.asarray(x).ravel()


def gaussian_kde_1d(x_samples, x_grid, bandwidth):
    """
    简单的一维高斯核密度估计（不依赖 scipy）。

    :param x_samples: (N,) 样本点
    :param x_grid: (M,) 网格点
    :param bandwidth: 核带宽（与 x 同单位）
    :return: (M,) 在 x_grid 上的密度（未归一化的“数密度”量纲）
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth 必须为正数")
    x_samples = np.asarray(x_samples)
    x_grid = np.asarray(x_grid)
    # 逐点累加高斯核
    diff = (x_grid[None, :] - x_samples[:, None]) / bandwidth
    density = np.exp(-0.5 * diff * diff).sum(axis=0)
    return density


def main():
    parser = argparse.ArgumentParser(description="根据粒子 x 位置绘制数密度分布曲线（第一列为 x）")
    parser.add_argument('--file', '-f', type=str, required=False, help='输入文件路径（.csv/.txt/.npy），第一列为粒子 x；若不提供，则使用内置数据')
    parser.add_argument('--column', '-c', type=int, default=0, help='作为 x 的列索引（默认 0）')
    parser.add_argument('--xmax', type=float, default=0.034, help='x 轴最大值（单位 m，默认 0.034）')
    parser.add_argument('--grid', type=int, default=200, help='x 网格数量（默认 200）')
    parser.add_argument('--bandwidth', type=float, default=1e-3, help='核带宽（单位 m，默认 1e-3 = 1 mm）')
    parser.add_argument('--hist', action='store_true', help='是否叠加直方图作为参考')
    parser.add_argument('--save', type=str, default=None, help='保存图片到此路径（若不指定则直接显示）')

    args = parser.parse_args()

    if args.file:
        x = load_x_positions(args.file, column_index=args.column)
    else:
        # 使用内置数据
        x = EMBEDDED_POSITIONS[:, 0]

    # 自动估计 x 轴范围
    xmin = max(0.0, float(np.nanmin(x)) if np.isfinite(x).any() else 0.0)
    xmax = args.xmax if args.xmax is not None else float(np.nanmax(x))
    if not np.isfinite(xmax) or xmax <= xmin:
        xmax = 0.034

    x_grid = np.linspace(xmin, xmax, args.grid)
    density = gaussian_kde_1d(x, x_grid, bandwidth=args.bandwidth)

    # 绘图（x 轴转换为 mm，更易读）
    x_grid_mm = x_grid * 1000.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_grid_mm, density, 'b-', linewidth=2, label='Number density (KDE, a.u.)')

    if args.hist:
        ax.hist(x * 1000.0, bins=max(10, args.grid // 10), density=False, alpha=0.3, color='gray', label='Histogram (counts)')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Density (arb. units)')
    ax.set_title('Particle number density along x')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()

