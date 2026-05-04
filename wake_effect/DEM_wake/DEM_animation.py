import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from pathlib import Path

from DEM_wake import (
    WAKE_CLOSURE_RE_DEFAULT,
    _A_P,
    acoustic_wake_velocity,
    particle_reynolds_diameter,
)

# =========================
# 参数设置（与 DEM_wake.py：ρ=1、μ=1e-3、直径 1 mm、|U|=1 m/s、Re_D=1 一致）
# =========================
dt = 0.02
steps = 300

particle_pos0 = np.array([0.008, 0.0])
particle_vel = np.array([-1.0, 0.0])
U_REF = float(np.linalg.norm(particle_vel))  # 粒子速度模长 [m/s]，勿与 quiver 初值数组同名
# 粒子雷诺数 Re_D = |U|D/ν（直径 D）；闭式参量见 DEM_wake 模块说明
RE = particle_reynolds_diameter(U_REF)
re_closure = WAKE_CLOSURE_RE_DEFAULT  # 传入 acoustic_wake_velocity(..., Re=...) 以保持当前正确结果
STATIC_OUT = Path(__file__).resolve().with_name("DEM_animation_snapshot.png")

def build_polar_ring_points(radius, n_rings, center=(0.0, 0.0)):
    """分圈采样：半径越大每圈点数越多，并做逐圈角度偏移。"""
    if n_rings < 2:
        raise ValueError("n_rings 必须 >= 2")
    cx, cy = center
    r_samples = np.linspace(radius / n_rings, radius, n_rings)
    x_points = [cx]
    y_points = [cy]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i, r in enumerate(r_samples, start=1):
        frac = i / n_rings
        n_theta = max(8, int(round(n_rings * (0.6 + 1.8 * frac))))
        theta_offset = (i * golden_angle) % (2.0 * np.pi)
        theta_ring = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + theta_offset
        x_points.extend((cx + r * np.cos(theta_ring)).tolist())
        y_points.extend((cy + r * np.sin(theta_ring)).tolist())
    return np.asarray(x_points), np.asarray(y_points)


# 采样设置（用于流场；物理量仍为米，仅显示轴以粒子半径 a=_A_P 无量纲化）
grid_size = 18
DOMAIN_RADIUS = 5e-3  # 考察半径 [m]（粒子系）
X, Y = build_polar_ring_points(DOMAIN_RADIUS, grid_size)
# 粒子圆盘 r<=_A_P 内的采样点不画箭头（与下方 Circle 一致，粒子系原点在圆心）
_mask_outside = np.hypot(X, Y) > _A_P
X = X[_mask_outside]
Y = Y[_mask_outside]
local_pts = np.column_stack([X, Y])
# 坐标轴以「几倍粒子半径」为单位：x_R = x/a, y_R = y/a
X_R = X / _A_P
Y_R = Y / _A_P
DOMAIN_HALF_R = DOMAIN_RADIUS / _A_P  # 视域半宽 [×a]

# =========================
# 计算流场
# =========================
def compute_flow_field(p_pos, p_vel):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for k in range(X.size):
        target = p_pos + local_pts[k]
        vx, vy = acoustic_wake_velocity(
            source_pos=p_pos,
            target_pos=target,
            source_vel=p_vel,
            Re=re_closure,
        )
        U[k] = vx
        V[k] = vy
    return U, V


# =========================
# 初始化画布
# =========================
fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

U_init = np.zeros_like(X)
V_init = np.zeros_like(Y)
speed0 = np.zeros_like(X)

# 归一化箭头方向，颜色映射速度大小
cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=0, vmax=max(U_REF * 1e-6, 1e-9))

# ARROW_SCALE 为米制显示长度；轴为 x/a 时箭头分量需除以 a 以匹配 scale_units='xy'
ARROW_SCALE = 1.75e-4
ARROW_SCALE_R = ARROW_SCALE / _A_P

quiver = ax.quiver(
    X_R, Y_R, U_init, V_init, speed0,
    cmap=cmap, norm=norm,
    scale=1,
    scale_units='xy',    # 箭头长度直接用数据单位
    width=0.0018,        # 箭头粗细（axes fraction）
    headwidth=5,
    headlength=5,
    headaxislength=4,
    alpha=0.9,
    minlength=0.1,
)

cbar = fig.colorbar(quiver, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Flow Speed (m/s)", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# 粒子在粒子系原点；在 x/a、y/a 坐标下半径为 1（即 1 倍粒子半径）
particle_patch = Circle(
    (0.0, 0.0),
    radius=1.0,
    facecolor="#00e5ff",
    edgecolor="white",
    linewidth=0.8,
    zorder=5,
)
ax.add_patch(particle_patch)

ax.set_xlim(-DOMAIN_HALF_R, DOMAIN_HALF_R)
ax.set_ylim(-DOMAIN_HALF_R, DOMAIN_HALF_R)
ax.set_aspect("equal")
ax.set_xlabel(r"$x\,/\,a$", color="white", fontsize=11)
ax.set_ylabel(r"$y\,/\,a$", color="white", fontsize=11)
ax.set_title(
    f"Particle Wake in Water (particle frame, Re={RE:.0f}, |U|={U_REF:g} m/s)",
    color="white",
    fontsize=13,
    pad=12,
)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444444')

time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                    color='white', fontsize=10, va='top')


# =========================
# 更新函数（动画核心）
# =========================
def update(frame):
    particle_pos = particle_pos0 + particle_vel * dt * frame

    U, V = compute_flow_field(particle_pos, particle_vel)
    speed = np.sqrt(U**2 + V**2)

    # 归一化方向向量，乘以固定显示比例，保证箭头长度统一且适中
    eps = 1e-12
    U_norm = U / (speed + eps) * ARROW_SCALE_R
    V_norm = V / (speed + eps) * ARROW_SCALE_R

    quiver.set_UVC(U_norm, V_norm, speed)
    s_max = float(np.percentile(speed, 99.0)) if np.any(speed > 0) else max(U_REF * 1e-9, 1e-12)
    norm.vmax = max(s_max, max(U_REF * 1e-9, 1e-12))

    time_text.set_text(f"t = {frame * dt:.2f} s\nframe: particle-centered")

    return quiver, particle_patch, time_text


# =========================
# 动画
# =========================
update(0)
plt.tight_layout()
fig.savefig(STATIC_OUT, dpi=220, facecolor=fig.get_facecolor())
print(f"Saved static: {STATIC_OUT.name}")

anim = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=50,
    blit=False,
    repeat=True
)

plt.show()