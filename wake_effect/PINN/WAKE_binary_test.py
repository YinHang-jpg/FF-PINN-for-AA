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
CHAR_LENGTH_M = 5e-4

# 导入 DEM 数值计算
WAKE_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'DEM_wake')
if WAKE_DIR not in sys.path:
    sys.path.insert(0, WAKE_DIR)
from DEM_wake import acoustic_wake_velocity


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
        uv[i] = acoustic_wake_velocity(source_pos, targets[i], source_vel, Re)
    return uv


def wake_velocity_pinn(source_pos, targets):
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

    cos_th = np.cos(Th_np);  sin_th = np.sin(Th_np)
    U_v = vr * cos_th - vt * sin_th
    V_v = vr * sin_th + vt * cos_th

    x_v = x_rel[valid]
    U_abs = np.abs(U_v)
    U_v = np.where(x_v > 0, -U_abs, -U_abs * 0.5)

    U_v = np.where(np.isfinite(U_v), U_v, 0.0)
    V_v = np.where(np.isfinite(V_v), V_v, 0.0)
    U_out[valid] = U_v;  V_out[valid] = V_v
    return np.column_stack([U_out, V_out])


# ============ 物理参数 ============
Re = 1.0
viscosity = 1.8e-5
d_p = 2e-6
C_c = 1.083
rho_p = 2000.0
r_p = d_p / 2.0
mass_p = rho_p * 4.0 / 3.0 * np.pi * r_p ** 3
drag_coeff = 3.0 * np.pi * viscosity * d_p / C_c
tau = mass_p / drag_coeff  # Stokes 松弛时间

# ============ 场景设置 ============
dt = 0.02
steps = 300

source_pos0 = np.array([0.008, 0.0])
source_vel = np.array([-0.001, 0.0])

N_passive = 30
rng = np.random.default_rng(42)
passive_init = rng.uniform(-0.008, 0.008, (N_passive, 2))

pos_dem = passive_init.copy();  vel_dem = np.zeros((N_passive, 2))
pos_pinn = passive_init.copy(); vel_pinn = np.zeros((N_passive, 2))

# quiver 网格
grid_size = 14
xg = np.linspace(-0.01, 0.01, grid_size)
yg = np.linspace(-0.01, 0.01, grid_size)
XG, YG = np.meshgrid(xg, yg)
grid_pts = np.column_stack([XG.ravel(), YG.ravel()])

# ============ 画布 ============
fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(15, 7), facecolor="#0d1117")
for ax in (ax_d, ax_p):
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-0.01, 0.01);  ax.set_ylim(-0.01, 0.01)
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444444')

ax_d.set_title("DEM (Numerical)", color="white", fontsize=13, pad=10)
ax_p.set_title("PINN (Model)", color="white", fontsize=13, pad=10)

cmap = plt.cm.plasma
norm_c = mcolors.Normalize(vmin=0, vmax=0.05)
ARROW_SCALE = 5e-4
U0 = np.zeros_like(XG);  V0 = np.zeros_like(YG);  S0 = np.zeros_like(XG)

qkw = dict(cmap=cmap, norm=norm_c, scale=1, scale_units='xy',
           width=0.0018, headwidth=5, headlength=5, headaxislength=4,
           alpha=0.7, minlength=0.1)
quiver_d = ax_d.quiver(XG, YG, U0, V0, S0, **qkw)
quiver_p = ax_p.quiver(XG, YG, U0, V0, S0, **qkw)

for qv, ax in [(quiver_d, ax_d), (quiver_p, ax_p)]:
    cb = fig.colorbar(qv, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Flow Speed", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

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

def _update_quiver(source_pos, method):
    """计算 quiver 场"""
    if method == 'dem':
        uv = wake_velocity_dem(source_pos, source_vel, grid_pts)
    else:
        uv = wake_velocity_pinn(source_pos, grid_pts)
    U = uv[:, 0].reshape(XG.shape)
    V = uv[:, 1].reshape(YG.shape)
    speed = np.sqrt(U ** 2 + V ** 2)
    eps = 1e-12
    Un = U / (speed + eps) * ARROW_SCALE
    Vn = V / (speed + eps) * ARROW_SCALE
    return Un, Vn, speed


def _step_passive(pos, vel, source_pos, method):
    """用 Stokes 阻力推进被动粒子一步（解析积分，精确稳定）"""
    if method == 'dem':
        wake_uv = wake_velocity_dem(source_pos, source_vel, pos)
    else:
        wake_uv = wake_velocity_pinn(source_pos, pos)

    # 解析 Stokes 积分: v' = u + (v - u)*exp(-dt/tau), x' = x + u*dt + (v-u)*tau*(1-exp(-dt/tau))
    exp_f = np.exp(-dt / tau)
    dv = vel - wake_uv
    pos_new = pos + wake_uv * dt + dv * tau * (1.0 - exp_f)
    vel_new = wake_uv + dv * exp_f
    return pos_new, vel_new


def update(frame):
    global pos_dem, vel_dem, pos_pinn, vel_pinn

    source_pos = source_pos0 + source_vel * dt * frame

    # DEM 侧
    Ud, Vd, Sd = _update_quiver(source_pos, 'dem')
    quiver_d.set_UVC(Ud, Vd, Sd)
    pos_dem, vel_dem = _step_passive(pos_dem, vel_dem, source_pos, 'dem')
    src_d.set_data([source_pos[0]], [source_pos[1]])
    pas_d.set_data(pos_dem[:, 0], pos_dem[:, 1])
    txt_d.set_text(f"t = {frame * dt:.2f} s\npassive: {N_passive}")

    # PINN 侧
    Up, Vp, Sp = _update_quiver(source_pos, 'pinn')
    quiver_p.set_UVC(Up, Vp, Sp)
    pos_pinn, vel_pinn = _step_passive(pos_pinn, vel_pinn, source_pos, 'pinn')
    src_p.set_data([source_pos[0]], [source_pos[1]])
    pas_p.set_data(pos_pinn[:, 0], pos_pinn[:, 1])
    txt_p.set_text(f"t = {frame * dt:.2f} s\npassive: {N_passive}")

    return (quiver_d, quiver_p, src_d, src_p, pas_d, pas_p, txt_d, txt_p)


anim = FuncAnimation(fig, update, frames=steps, interval=50, blit=False, repeat=True)
plt.tight_layout()
plt.show()
