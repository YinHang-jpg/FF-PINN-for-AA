import numpy as np
import torch
import torch.nn as nn
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_OUT = os.path.join(SCRIPT_DIR, "WAKE_binary_test_snapshot.png")

# 导入 DEM 数值计算
WAKE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'DEM_wake')
if WAKE_DIR not in sys.path:
    sys.path.insert(0, WAKE_DIR)
from DEM_wake import CHAR_LENGTH_M, WAKE_CLOSURE_RE_DEFAULT, acoustic_wake_velocity


# ============ PINN 网络定义 ============

class WakeNetVrPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.layers = nn.Sequential(
            nn.Linear(1 + 2 * n_harmonics, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, R_s, Th):
        p = self.harmonics * Th
        return self.layers(torch.cat([R_s, torch.sin(p), torch.cos(p)], 1))


class WakeNetVtPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.layers = nn.Sequential(
            nn.Linear(1 + 2 * n_harmonics, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, R_s, Th):
        p = self.harmonics * Th
        return self.layers(torch.cat([R_s, torch.sin(p), torch.cos(p)], 1))


class WakeNetVrOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.layers = nn.Sequential(
            nn.Linear(1 + 2 * n_harmonics, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, R_s, Th):
        p = self.harmonics * Th
        return self.layers(torch.cat([R_s, torch.sin(p), torch.cos(p)], 1))


class WakeNetVtOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        self.layers = nn.Sequential(
            nn.Linear(1 + 2 * n_harmonics, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, R_s, Th):
        p = self.harmonics * Th
        return self.layers(torch.cat([R_s, torch.sin(p), torch.cos(p)], 1))


# ============ 加载 PINN 模型 ============

def _load(cls, pth, jf, dev):
    with open(os.path.join(SCRIPT_DIR, jf)) as f:
        par = json.load(f)
    m = cls()
    m.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, pth), map_location=dev, weights_only=True))
    m.to(dev); m.eval()
    return m, par

device = torch.device('cpu')
m_VrPP, p_VrPP = _load(WakeNetVrPP, 'wake_Vr_PP.pth', 'wake_Vr_PP_norm.json', device)
m_VtPP, p_VtPP = _load(WakeNetVtPP, 'wake_Vt_PP.pth', 'wake_Vt_PP_norm.json', device)
m_VrOs, p_VrOs = _load(WakeNetVrOseen, 'wake_Vr_Oseen.pth', 'wake_Vr_Oseen_norm.json', device)
m_VtOs, p_VtOs = _load(WakeNetVtOseen, 'wake_Vt_Oseen.pth', 'wake_Vt_Oseen_norm.json', device)
print("PINN 模型加载完成。")


def _pinn_pred(model, par, R_t, Th_t):
    R_s = (R_t - par['R_min']) / (par['R_max'] - par['R_min'])
    with torch.no_grad():
        return (model(R_s, Th_t) * par['force_sigma'] + par['force_mu']).cpu().numpy().flatten()


# ============ 批量 wake 速度计算 ============

def wake_velocity_dem(source_pos, source_vel, targets):
    """DEM 数值公式：逐点调用 acoustic_wake_velocity"""
    N = len(targets)
    uv = np.zeros((N, 2))
    for i in range(N):
        uv[i] = acoustic_wake_velocity(source_pos, targets[i], source_vel, RE_CLOSURE)
    return uv


def wake_velocity_pinn(source_pos, source_vel, targets):
    """PINN 模型批量推理，拼接方式与 DEM_wake.py 一致"""
    x_rel = targets[:, 0] - source_pos[0]
    y_rel = targets[:, 1] - source_pos[1]
    r = np.hypot(x_rel, y_rel)

    U_out = np.zeros(len(targets))
    V_out = np.zeros(len(targets))

    valid = r > 1e-12
    if not np.any(valid):
        return np.column_stack([U_out, V_out])

    R_np = r[valid] / CHAR_LENGTH_M
    Th_np = np.arctan2(y_rel[valid], x_rel[valid])
    R_t = torch.tensor(R_np.reshape(-1, 1), dtype=torch.float32)
    Th_t = torch.tensor(Th_np.reshape(-1, 1), dtype=torch.float32)

    vrPP = _pinn_pred(m_VrPP, p_VrPP, R_t, Th_t)
    vtPP = _pinn_pred(m_VtPP, p_VtPP, R_t, Th_t)
    vrOs = _pinn_pred(m_VrOs, p_VrOs, R_t, Th_t)
    vtOs = _pinn_pred(m_VtOs, p_VtOs, R_t, Th_t)

    pp = R_np < 2;  bl = (R_np >= 2) & (R_np <= 5);  os_ = R_np > 5
    vr = np.zeros_like(R_np);  vt = np.zeros_like(R_np)
    vr[pp] = vrPP[pp];  vt[pp] = vtPP[pp]
    vr[os_] = vrOs[os_];  vt[os_] = vtOs[os_]
    Rb = R_np[bl]
    vr[bl] = ((5 - Rb) * vrPP[bl] + (Rb - 2) * vrOs[bl]) / 3
    vt[bl] = ((5 - Rb) * vtPP[bl] + (Rb - 2) * vtOs[bl]) / 3

    cos_th = np.cos(Th_np)
    sin_th = np.sin(Th_np)
    U_v = vr * cos_th - vt * sin_th
    V_v = vr * sin_th + vt * cos_th

    # 无量纲 → m/s（与 DEM_wake.acoustic_wake_velocity；勿对 vx 做半平面修正）
    source_speed = np.linalg.norm(source_vel)
    U_v = U_v * source_speed
    V_v = V_v * source_speed

    U_v = np.where(np.isfinite(U_v), U_v, 0.0)
    V_v = np.where(np.isfinite(V_v), V_v, 0.0)
    U_out[valid] = U_v;  V_out[valid] = V_v
    return np.column_stack([U_out, V_out])


# ============ 物理参数 ============
RE_CLOSURE = WAKE_CLOSURE_RE_DEFAULT  # 无量纲闭式/PINN 参量（见 DEM_wake）；勿与粒子雷诺数 Re 混名
viscosity = 1.8e-5
d_p = 2e-6
C_c = 1.083
rho_p = 2000.0
r_p = d_p / 2.0
mass_p = rho_p * 4.0 / 3.0 * np.pi * r_p ** 3
drag_coeff = 3.0 * np.pi * viscosity * d_p / C_c
tau = mass_p / drag_coeff  # Stokes 松弛时间

# ============ 场景设置 ============
dt = 0.01
steps = 300

source_pos0 = np.array([0.008, 0.0])
source_vel = np.array([-0.01, 0.0])

# 被动粒子改为规则矩形阵列（替代随机分布）
N_COLS = 6
N_ROWS = 5
x_passive = np.linspace(-0.007, 0.007, N_COLS)
y_passive = np.linspace(-0.006, 0.006, N_ROWS)
XP, YP = np.meshgrid(x_passive, y_passive)
passive_init = np.column_stack([XP.ravel(), YP.ravel()])
N_passive = passive_init.shape[0]

pos_dem = passive_init.copy();  vel_dem = np.zeros((N_passive, 2))
pos_pinn = passive_init.copy(); vel_pinn = np.zeros((N_passive, 2))

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


# quiver 采样点（分圈递增密度）
grid_size = 14
DOMAIN_RADIUS = 20e-6
XG, YG = build_polar_ring_points(DOMAIN_RADIUS, grid_size)
grid_pts = np.column_stack([XG, YG])

# ============ 画布 ============
fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(15, 7), facecolor="#0d1117")
for ax in (ax_d, ax_p):
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-DOMAIN_RADIUS, DOMAIN_RADIUS);  ax.set_ylim(-DOMAIN_RADIUS, DOMAIN_RADIUS)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')

ax_d.set_title("DEM (Numerical)", color="white", fontsize=13, pad=10)
ax_p.set_title("PINN (Model)", color="white", fontsize=13, pad=10)

cmap = plt.cm.plasma
norm_d = mcolors.Normalize(vmin=0, vmax=1e-6)
norm_p = mcolors.Normalize(vmin=0, vmax=1e-6)
ARROW_SCALE = 7e-7
U0 = np.zeros_like(XG);  V0 = np.zeros_like(YG);  S0 = np.zeros_like(XG)

qkw_base = dict(cmap=cmap, scale=1, scale_units='xy',
                width=0.0018, headwidth=5, headlength=5, headaxislength=4,
                alpha=0.7, minlength=0.1)
quiver_d = ax_d.quiver(XG, YG, U0, V0, S0, norm=norm_d, **qkw_base)
quiver_p = ax_p.quiver(XG, YG, U0, V0, S0, norm=norm_p, **qkw_base)

cb_d = fig.colorbar(quiver_d, ax=ax_d, fraction=0.046, pad=0.04)
cb_d.set_label("Flow Speed", color="white", fontsize=9)
cb_d.ax.yaxis.set_tick_params(color="white")
plt.setp(cb_d.ax.yaxis.get_ticklabels(), color="white")

cb_p = fig.colorbar(quiver_p, ax=ax_p, fraction=0.046, pad=0.04)
cb_p.set_label("Flow Speed", color="white", fontsize=9)
cb_p.ax.yaxis.set_tick_params(color="white")
plt.setp(cb_p.ax.yaxis.get_ticklabels(), color="white")

src_d, = ax_d.plot([], [], 'o', color='#00e5ff', ms=10,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=5)
src_p, = ax_p.plot([], [], 'o', color='#00e5ff', ms=10,
                   markeredgecolor='white', markeredgewidth=0.8, zorder=5)
pas_d, = ax_d.plot([], [], 'o', color='#ff6ec7', ms=5,
                   markeredgecolor='white', markeredgewidth=0.4, zorder=4)
pas_p, = ax_p.plot([], [], 'o', color='#ff6ec7', ms=5,
                   markeredgecolor='white', markeredgewidth=0.4, zorder=4)

txt_d = ax_d.text(0.02, 0.96, '', transform=ax_d.transAxes,
                  color='white', fontsize=9, va='top')
txt_p = ax_p.text(0.02, 0.96, '', transform=ax_p.transAxes,
                  color='white', fontsize=9, va='top')


# ============ 更新函数 ============

def _update_quiver(source_pos, source_vel, method):
    """计算 quiver 场"""
    targets = grid_pts + source_pos
    if method == 'dem':
        uv = wake_velocity_dem(source_pos, source_vel, targets)
    else:
        uv = wake_velocity_pinn(source_pos, source_vel, targets)
    U = uv[:, 0]
    V = uv[:, 1]
    speed = np.sqrt(U ** 2 + V ** 2)
    eps = 1e-12
    Un = U / (speed + eps) * ARROW_SCALE
    Vn = V / (speed + eps) * ARROW_SCALE
    return Un, Vn, speed


def _step_passive(pos, vel, source_pos, source_vel, method):
    """用 Stokes 阻力推进被动粒子一步（解析积分，精确稳定）"""
    if method == 'dem':
        wake_uv = wake_velocity_dem(source_pos, source_vel, pos)
    else:
        wake_uv = wake_velocity_pinn(source_pos, source_vel, pos)

    # 解析 Stokes 积分: v' = u + (v - u)*exp(-dt/tau), x' = x + u*dt + (v-u)*tau*(1-exp(-dt/tau))
    exp_f = np.exp(-dt / tau)
    dv = vel - wake_uv
    pos_new = pos + wake_uv * dt + dv * tau * (1.0 - exp_f)
    vel_new = wake_uv + dv * exp_f
    return pos_new, vel_new


def export_simulation_json():
    """导出 0.00s~0.10s (步长 0.01s) 的粒子轨迹，重复运行时覆盖同名文件。"""
    export_path = os.path.join(SCRIPT_DIR, "wake_binary_test_positions.json")
    sample_times = np.round(np.linspace(0.0, 0.1, 11), 2)

    pos_dem_exp = passive_init.copy()
    vel_dem_exp = np.zeros((N_passive, 2))
    pos_pinn_exp = passive_init.copy()
    vel_pinn_exp = np.zeros((N_passive, 2))
    prev_t = 0.0

    records = []
    for t in sample_times:
        delta_t = t - prev_t
        if delta_t > 0.0:
            exp_f = np.exp(-delta_t / tau)
            source_pos_prev = source_pos0 + source_vel * prev_t

            wake_dem = wake_velocity_dem(source_pos_prev, source_vel, pos_dem_exp)
            dv_dem = vel_dem_exp - wake_dem
            pos_dem_exp = pos_dem_exp + wake_dem * delta_t + dv_dem * tau * (1.0 - exp_f)
            vel_dem_exp = wake_dem + dv_dem * exp_f

            wake_pinn = wake_velocity_pinn(source_pos_prev, source_vel, pos_pinn_exp)
            dv_pinn = vel_pinn_exp - wake_pinn
            pos_pinn_exp = pos_pinn_exp + wake_pinn * delta_t + dv_pinn * tau * (1.0 - exp_f)
            vel_pinn_exp = wake_pinn + dv_pinn * exp_f

        records.append({
            "time_s": float(t),
            "dem_particles_xy": pos_dem_exp.tolist(),
            "pinn_particles_xy": pos_pinn_exp.tolist()
        })
        prev_t = t

    payload = {
        "source_particle_initial_position_xy_m": source_pos0.tolist(),
        "source_particle_initial_velocity_xy_mps": source_vel.tolist(),
        "passive_particles_initial_positions_xy_m": passive_init.tolist(),
        "time_series": records
    }

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已导出 JSON: {export_path}")


def update(frame):
    global pos_dem, vel_dem, pos_pinn, vel_pinn

    source_pos = source_pos0 + source_vel * dt * frame

    # DEM 侧
    Ud, Vd, Sd = _update_quiver(source_pos, source_vel, 'dem')
    quiver_d.set_UVC(Ud, Vd, Sd)
    s_max_d = Sd.max() if Sd.max() > 1e-30 else 1e-6
    norm_d.vmax = s_max_d
    pos_dem, vel_dem = _step_passive(pos_dem, vel_dem, source_pos, source_vel, 'dem')
    src_d.set_data([0.0], [0.0])
    rel_dem = pos_dem - source_pos
    pas_d.set_data(rel_dem[:, 0], rel_dem[:, 1])
    txt_d.set_text(f"t = {frame * dt:.2f} s\npassive: {N_passive}\nframe: particle-centered")

    # PINN 侧
    Up, Vp, Sp = _update_quiver(source_pos, source_vel, 'pinn')
    quiver_p.set_UVC(Up, Vp, Sp)
    s_max_p = Sp.max() if Sp.max() > 1e-30 else 1e-6
    norm_p.vmax = s_max_p
    pos_pinn, vel_pinn = _step_passive(pos_pinn, vel_pinn, source_pos, source_vel, 'pinn')
    src_p.set_data([0.0], [0.0])
    rel_pinn = pos_pinn - source_pos
    pas_p.set_data(rel_pinn[:, 0], rel_pinn[:, 1])
    txt_p.set_text(f"t = {frame * dt:.2f} s\npassive: {N_passive}\nframe: particle-centered")

    return (quiver_d, quiver_p, src_d, src_p, pas_d, pas_p, txt_d, txt_p)


def render_static_snapshot():
    """渲染并保存粒子参考系下的初始静态图（覆盖同名文件）。"""
    source_pos = source_pos0.copy()

    Ud, Vd, Sd = _update_quiver(source_pos, source_vel, 'dem')
    quiver_d.set_UVC(Ud, Vd, Sd)
    s_max_d = Sd.max() if Sd.max() > 1e-30 else 1e-6
    norm_d.vmax = s_max_d
    src_d.set_data([0.0], [0.0])
    rel_dem0 = passive_init - source_pos
    pas_d.set_data(rel_dem0[:, 0], rel_dem0[:, 1])
    txt_d.set_text(f"t = 0.00 s\npassive: {N_passive}\nframe: particle-centered")

    Up, Vp, Sp = _update_quiver(source_pos, source_vel, 'pinn')
    quiver_p.set_UVC(Up, Vp, Sp)
    s_max_p = Sp.max() if Sp.max() > 1e-30 else 1e-6
    norm_p.vmax = s_max_p
    src_p.set_data([0.0], [0.0])
    rel_pinn0 = passive_init - source_pos
    pas_p.set_data(rel_pinn0[:, 0], rel_pinn0[:, 1])
    txt_p.set_text(f"t = 0.00 s\npassive: {N_passive}\nframe: particle-centered")

    plt.tight_layout()
    fig.savefig(STATIC_OUT, dpi=220, facecolor=fig.get_facecolor())
    print("Saved static: WAKE_binary_test_snapshot.png")


export_simulation_json()
render_static_snapshot()
anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=True)
plt.show()
