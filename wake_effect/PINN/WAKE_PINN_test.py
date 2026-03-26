import numpy as np
import torch
import torch.nn as nn
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAR_LENGTH_M = 5e-4


# ============ 网络定义（必须与训练时结构完全一致） ============

class WakeNetVrPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        input_dim = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, R_scaled, Theta):
        phases = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.layers(x)


class WakeNetVtPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        input_dim = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, R_scaled, Theta):
        phases = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.layers(x)


class WakeNetVrOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        input_dim = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, R_scaled, Theta):
        phases = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.layers(x)


class WakeNetVtOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        input_dim = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, R_scaled, Theta):
        phases = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(phases), torch.cos(phases)], dim=1)
        return self.layers(x)


# ============ 加载模型 ============

def load_model_and_params(model_class, pth_file, json_file, device):
    with open(os.path.join(SCRIPT_DIR, json_file), 'r') as f:
        params = json.load(f)
    model = model_class()
    model.load_state_dict(
        torch.load(os.path.join(SCRIPT_DIR, pth_file), map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, params


device = torch.device('cpu')

model_Vr_PP, p_Vr_PP = load_model_and_params(
    WakeNetVrPP, 'wake_Vr_PP.pth', 'wake_Vr_PP_norm.json', device)
model_Vt_PP, p_Vt_PP = load_model_and_params(
    WakeNetVtPP, 'wake_Vt_PP.pth', 'wake_Vt_PP_norm.json', device)
model_Vr_Oseen, p_Vr_Oseen = load_model_and_params(
    WakeNetVrOseen, 'wake_Vr_Oseen.pth', 'wake_Vr_Oseen_norm.json', device)
model_Vt_Oseen, p_Vt_Oseen = load_model_and_params(
    WakeNetVtOseen, 'wake_Vt_Oseen.pth', 'wake_Vt_Oseen_norm.json', device)

print("4 个 PINN 模型全部加载完成。")


def pinn_predict(model, params, R_tensor, Theta_tensor):
    R_scaled = (R_tensor - params['R_min']) / (params['R_max'] - params['R_min'])
    with torch.no_grad():
        pred_norm = model(R_scaled, Theta_tensor)
    return pred_norm * params['force_sigma'] + params['force_mu']


# ============ 批量计算流场（向量化，一次推理所有网格点） ============

def compute_flow_field_pinn(p_pos):
    x_rel = X.flatten() - p_pos[0]
    y_rel = Y.flatten() - p_pos[1]
    r = np.hypot(x_rel, y_rel)

    valid = r > 1e-12
    R_np = r[valid] / CHAR_LENGTH_M
    Theta_np = np.arctan2(y_rel[valid], x_rel[valid])

    R_t = torch.tensor(R_np.reshape(-1, 1), dtype=torch.float32, device=device)
    Theta_t = torch.tensor(Theta_np.reshape(-1, 1), dtype=torch.float32, device=device)

    # 4 个 PINN 批量推理
    Vr_PP_t = pinn_predict(model_Vr_PP, p_Vr_PP, R_t, Theta_t).cpu().numpy().flatten()
    Vt_PP_t = pinn_predict(model_Vt_PP, p_Vt_PP, R_t, Theta_t).cpu().numpy().flatten()
    Vr_Os_t = pinn_predict(model_Vr_Oseen, p_Vr_Oseen, R_t, Theta_t).cpu().numpy().flatten()
    Vt_Os_t = pinn_predict(model_Vt_Oseen, p_Vt_Oseen, R_t, Theta_t).cpu().numpy().flatten()

    # 分段混合（与 DEM_wake.py 完全一致）
    pp_mask = R_np < 2
    blend_mask = (R_np >= 2) & (R_np <= 5)
    oseen_mask = R_np > 5

    vr = np.zeros_like(R_np)
    vt = np.zeros_like(R_np)

    vr[pp_mask] = Vr_PP_t[pp_mask]
    vt[pp_mask] = Vt_PP_t[pp_mask]

    vr[oseen_mask] = Vr_Os_t[oseen_mask]
    vt[oseen_mask] = Vt_Os_t[oseen_mask]

    R_b = R_np[blend_mask]
    vr[blend_mask] = ((5 - R_b) * Vr_PP_t[blend_mask] + (R_b - 2) * Vr_Os_t[blend_mask]) / 3
    vt[blend_mask] = ((5 - R_b) * Vt_PP_t[blend_mask] + (R_b - 2) * Vt_Os_t[blend_mask]) / 3

    # 极坐标 → 笛卡尔
    cos_th = np.cos(Theta_np)
    sin_th = np.sin(Theta_np)
    U_valid = vr * cos_th - vt * sin_th
    V_valid = vr * sin_th + vt * cos_th

    # 方向调整（与 DEM_wake.py 一致）
    x_valid = x_rel[valid]
    U_abs = np.abs(U_valid)
    upstream = x_valid > 0
    U_valid = np.where(upstream, -U_abs, -U_abs * 0.5)

    # 填回完整网格
    U_flat = np.zeros(r.shape)
    V_flat = np.zeros(r.shape)
    U_flat[valid] = np.where(np.isfinite(U_valid), U_valid, 0.0)
    V_flat[valid] = np.where(np.isfinite(V_valid), V_valid, 0.0)

    return U_flat.reshape(X.shape), V_flat.reshape(Y.shape)


# ============ 动画（格式与 DEM_animation.py 完全相同） ============

Re = 1.0
dt = 0.02
steps = 300

particle_pos0 = np.array([0.008, 0.0])
particle_vel = np.array([-0.001, 0.0])

grid_size = 18
x_lin = np.linspace(-0.01, 0.01, grid_size)
y_lin = np.linspace(-0.01, 0.01, grid_size)
X, Y = np.meshgrid(x_lin, y_lin)

fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

U0 = np.zeros_like(X)
V0 = np.zeros_like(Y)
speed0 = np.zeros_like(X)

cmap = plt.cm.plasma
norm_color = mcolors.Normalize(vmin=0, vmax=0.05)
ARROW_SCALE = 5e-4

quiver = ax.quiver(
    X, Y, U0, V0, speed0,
    cmap=cmap, norm=norm_color,
    scale=1,
    scale_units='xy',
    width=0.0018,
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
ax.set_title("Particle Wake Flow Field (PINN)", color="white", fontsize=13, pad=12)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#444444')

time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes,
                    color='white', fontsize=10, va='top')


def update(frame):
    particle_pos = particle_pos0 + particle_vel * dt * frame

    U, V = compute_flow_field_pinn(particle_pos)
    speed = np.sqrt(U ** 2 + V ** 2)

    eps = 1e-12
    U_norm = U / (speed + eps) * ARROW_SCALE
    V_norm = V / (speed + eps) * ARROW_SCALE

    quiver.set_UVC(U_norm, V_norm, speed)

    particle_plot.set_data([particle_pos[0]], [particle_pos[1]])
    time_text.set_text(f"t = {frame * dt:.2f} s")

    return quiver, particle_plot, time_text


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
