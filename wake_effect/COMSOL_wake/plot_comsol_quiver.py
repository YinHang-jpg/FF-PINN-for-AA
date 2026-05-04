import argparse
import re
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
import numpy as np

# 与 DEM_wake 粒子半径一致（500 μm）
_DEM_WAKE_DIR = Path(__file__).resolve().parent.parent / "DEM_wake"
if str(_DEM_WAKE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEM_WAKE_DIR))
from DEM_wake import _A_P as PARTICLE_RADIUS_M


def load_comsol_table(path: Path) -> tuple[np.ndarray, float | None]:
    rows = []
    detected_time = None
    time_pattern = re.compile(r"@\s*t\s*=\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("%"):
                m = time_pattern.search(s)
                if m:
                    detected_time = float(m.group(1))
                continue
            rows.append([float(x) for x in s.split()])
    return np.asarray(rows, dtype=float), detected_time


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从 COMSOL_vx/vy 文本绘制规整矢量图。")
    parser.add_argument(
        "--grid",
        type=int,
        default=18,
        help="极坐标采样圈数（与 DEM_animation.py grid_size=18 相仿）",
    )
    parser.add_argument("--arrow-scale", type=float, default=1.75e-4, help="归一化箭头显示长度（与默认 5mm 视域匹配）")
    parser.add_argument(
        "--focus-radius-um",
        type=float,
        default=5000.0,
        help="聚焦半径（微米），以粒子中心(0,0)为中心，默认 5000 微米（5 mm）",
    )
    parser.add_argument(
        "--vx-file",
        type=Path,
        default=script_dir / "COMSOL_vx.txt",
        help="vx 数据文件路径",
    )
    parser.add_argument(
        "--vy-file",
        type=Path,
        default=script_dir / "COMSOL_vy.txt",
        help="vy 数据文件路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出图片路径（默认按时间自动命名）",
    )
    args = parser.parse_args()

    vx, t_vx = load_comsol_table(args.vx_file)
    vy, t_vy = load_comsol_table(args.vy_file)

    if vx.shape != vy.shape:
        raise RuntimeError(f"vx/vy 形状不一致: {vx.shape} vs {vy.shape}")
    if not np.allclose(vx[:, :2], vy[:, :2], atol=1e-12):
        raise RuntimeError("vx/vy 的 X/Y 坐标不一致，无法合并。")

    if vx.shape[1] < 3:
        raise RuntimeError(f"数据列不足，预期至少 3 列(X/Y/值)，实际为 {vx.shape[1]} 列。")

    # 由于上游导出仅保留最后一个时刻，统一读取最后一列。
    col = vx.shape[1] - 1
    final_time = t_vx if t_vx is not None else t_vy

    x = vx[:, 0]
    y = vx[:, 1]
    u_nodes = vx[:, col]
    v_nodes = vy[:, col]

    focus_r_m = args.focus_radius_um * 1e-6
    if focus_r_m <= 0:
        raise ValueError("--focus-radius-um 必须为正数。")

    if args.grid < 2:
        raise ValueError("--grid 必须 >= 2。")

    # 分圈采样：半径越大，每圈箭头数量越多；并加入逐圈角度偏移，避免沿固定极轴排布。
    r_samples = np.linspace(focus_r_m / args.grid, focus_r_m, args.grid)
    x_points = [0.0]
    y_points = [0.0]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i, r in enumerate(r_samples, start=1):
        frac = i / args.grid
        n_theta = max(8, int(round(args.grid * (0.6 + 1.8 * frac))))
        theta_offset = (i * golden_angle) % (2.0 * np.pi)
        theta_ring = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + theta_offset
        x_points.extend((r * np.cos(theta_ring)).tolist())
        y_points.extend((r * np.sin(theta_ring)).tolist())
    XG = np.asarray(x_points, dtype=float)
    YG = np.asarray(y_points, dtype=float)
    # 粒子圆盘内不画箭头（与 DEM_animation 一致）
    _outside = np.hypot(XG, YG) > PARTICLE_RADIUS_M
    XG = XG[_outside]
    YG = YG[_outside]

    tri = mtri.Triangulation(x, y)
    interp_u = mtri.LinearTriInterpolator(tri, u_nodes)
    interp_v = mtri.LinearTriInterpolator(tri, v_nodes)
    U = np.ma.filled(interp_u(XG, YG), 0.0)
    V = np.ma.filled(interp_v(XG, YG), 0.0)
    speed = np.hypot(U, V)

    eps = 1e-12
    U_norm = U / (speed + eps) * args.arrow_scale
    V_norm = V / (speed + eps) * args.arrow_scale

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    vmax = float(np.percentile(speed, 99.0)) if np.any(speed > 0) else 1e-6
    norm = mcolors.Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    q = ax.quiver(
        XG,
        YG,
        U_norm,
        V_norm,
        speed,
        cmap=plt.cm.plasma,
        norm=norm,
        scale=1,
        scale_units="xy",
        width=0.0018,
        headwidth=5,
        headlength=5,
        headaxislength=4,
        alpha=0.9,
        minlength=0.1,
    )

    cbar = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Flow Speed (m/s)", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    particle_patch = Circle(
        (0.0, 0.0),
        radius=PARTICLE_RADIUS_M,
        facecolor="#00e5ff",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(particle_patch)

    ax.set_xlim(-focus_r_m, focus_r_m)
    ax.set_ylim(-focus_r_m, focus_r_m)
    ax.set_aspect("equal")
    if final_time is None:
        title_time = "t=last"
    else:
        title_time = f"t={final_time:g} s"
    ax.set_title(
        f"COMSOL Wake ({title_time}, ±{args.focus_radius_um:g} μm, "
        f"particle r={PARTICLE_RADIUS_M * 1e6:.0f} μm)",
        color="white",
        fontsize=13,
        pad=12,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.text(
        0.02,
        0.96,
        f"polar rings={args.grid}, arrows={XG.size}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="top",
    )

    if args.out is None:
        if final_time is None:
            out_name = "COMSOL_wake_vector_tlast_regular.png"
        else:
            out_name = f"COMSOL_wake_vector_t{final_time:g}_regular.png"
        out_path = script_dir / out_name
    else:
        out_path = args.out
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()
