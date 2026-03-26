import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

from DEM_wake import acoustic_wake_velocity

# =========================
# 参数设置
# =========================
Re = 1.0
dt = 0.02
steps = 300

particle_pos0 = np.array([0.008, 0.0])
particle_vel = np.array([-0.001, 0.0])

# 网格设置（用于流场）
grid_size = 18
x = np.linspace(-0.01, 0.01, grid_size)
y = np.linspace(-0.01, 0.01, grid_size)
X, Y = np.meshgrid(x, y)

# =========================
# 计算流场
# =========================
def compute_flow_field(p_pos, p_vel):
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(grid_size):
        for j in range(grid_size):
            target = np.array([X[i, j], Y[i, j]])
            vx, vy = acoustic_wake_velocity(
                source_pos=p_pos,
                target_pos=target,
                source_vel=p_vel,
                Re=Re
            )
            U[i, j] = vx
            V[i, j] = vy
    return U, V


# =========================
# 初始化画布
# =========================
fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

U0 = np.zeros_like(X)
V0 = np.zeros_like(Y)
speed0 = np.zeros_like(X)

# 归一化箭头方向，颜色映射速度大小
cmap = plt.cm.plasma
norm = mcolors.Normalize(vmin=0, vmax=0.05)

# ARROW_SCALE 控制箭头显示长度（数据单位）
# 网格间距约 0.00118，箭头取间距的 45% ≈ 5e-4
ARROW_SCALE = 5e-4

quiver = ax.quiver(
    X, Y, U0, V0, speed0,
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
cbar.set_label("Flow Speed", color="white", fontsize=11)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

particle_plot, = ax.plot([], [], 'o',
                          color='#00e5ff', markersize=9,
                          markeredgecolor='white', markeredgewidth=0.8,
                          zorder=5)

ax.set_xlim(-0.01, 0.01)
ax.set_ylim(-0.01, 0.01)
ax.set_aspect('equal')
ax.set_title("Particle Wake Flow Field", color="white", fontsize=13, pad=12)
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
    U_norm = U / (speed + eps) * ARROW_SCALE
    V_norm = V / (speed + eps) * ARROW_SCALE

    quiver.set_UVC(U_norm, V_norm, speed)

    particle_plot.set_data([particle_pos[0]], [particle_pos[1]])
    time_text.set_text(f"t = {frame * dt:.2f} s")

    return quiver, particle_plot, time_text


# =========================
# 动画
# =========================
anim = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=50,
    blit=False,
    repeat=True
)

plt.tight_layout()
plt.show()