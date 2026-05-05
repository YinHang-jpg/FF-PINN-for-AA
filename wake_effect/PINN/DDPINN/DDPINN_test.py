"""DDPINN 测试与快照（与 wake_effect/PINN/WAKE_PINN_test.py 一致结构）。

加载 wake_effect/PINN/DDPINN/PINN/ 下的 4 个 DDPINN 权重，按 PP/Oseen 分段
混合得到无量纲极坐标速度，再合成笛卡尔速度并乘以 |U|（粒子速度模），
最终输出与 WAKE_PINN_test_snapshot.png 同样布局的分布图：DDPINN_test_snapshot.png。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PINN_DIR = os.path.dirname(SCRIPT_DIR)
WAKE_EFFECT_DIR = os.path.dirname(PINN_DIR)
_DEM_WAKE_DIR = os.path.join(WAKE_EFFECT_DIR, "DEM_wake")
if _DEM_WAKE_DIR not in sys.path:
    sys.path.insert(0, _DEM_WAKE_DIR)

from DEM_wake import (  # noqa: E402
    CHAR_LENGTH_M,
    WAKE_CLOSURE_RE_DEFAULT,
    _A_P,
)

STATIC_OUT = os.path.join(SCRIPT_DIR, "DDPINN_test_snapshot.png")

# DDPINN 权重目录：本脚本同级 PINN/ 子目录
_DDPINN_WEIGHT_DIR = os.path.join(SCRIPT_DIR, "PINN")


def _weight_dir() -> str:
    if os.path.isfile(os.path.join(_DDPINN_WEIGHT_DIR, "wake_Vr_PP.pth")):
        return _DDPINN_WEIGHT_DIR
    raise FileNotFoundError(
        f"未找到 DDPINN 权重: {_DDPINN_WEIGHT_DIR}; 请先运行 DDPINN一键训练.py"
    )


# ============ 网络定义（必须与训练时结构完全一致） ============

class WakeNetVrPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer("harmonics",
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
        self.register_buffer("harmonics",
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
        self.register_buffer("harmonics",
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
        self.register_buffer("harmonics",
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
    base = _weight_dir()
    with open(os.path.join(base, json_file), "r") as f:
        params = json.load(f)
    model = model_class()
    model.load_state_dict(
        torch.load(os.path.join(base, pth_file), map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, params


device = torch.device("cpu")

model_Vr_PP, p_Vr_PP = load_model_and_params(
    WakeNetVrPP, "wake_Vr_PP.pth", "wake_Vr_PP_norm.json", device)
model_Vt_PP, p_Vt_PP = load_model_and_params(
    WakeNetVtPP, "wake_Vt_PP.pth", "wake_Vt_PP_norm.json", device)
model_Vr_Oseen, p_Vr_Oseen = load_model_and_params(
    WakeNetVrOseen, "wake_Vr_Oseen.pth", "wake_Vr_Oseen_norm.json", device)
model_Vt_Oseen, p_Vt_Oseen = load_model_and_params(
    WakeNetVtOseen, "wake_Vt_Oseen.pth", "wake_Vt_Oseen_norm.json", device)

print(f"4 个 DDPINN 模型已从 {_weight_dir()} 加载完成。")


def pinn_predict(model, params, R_tensor, Theta_tensor):
    """与 PINN_wake/pinn_wake_infer.pinn_predict 一致：R 归一化并 clamp 到 [0,1]。"""
    R_scaled = (R_tensor - params["R_min"]) / (params["R_max"] - params["R_min"])
    R_scaled = torch.clamp(R_scaled, 0.0, 1.0)
    with torch.no_grad():
        pred_norm = model(R_scaled, Theta_tensor)
    return pred_norm * params["force_sigma"] + params["force_mu"]


# ============ 批量计算流场（向量化，一次推理所有网格点） ============

def compute_flow_field_ddpinn(p_pos, p_vel):
    x_rel = X
    y_rel = Y
    r = np.hypot(x_rel, y_rel)

    valid = r > 1e-12
    R_np = r[valid] / CHAR_LENGTH_M
    Theta_np = np.arctan2(y_rel[valid], x_rel[valid])

    R_t = torch.tensor(R_np.reshape(-1, 1), dtype=torch.float32, device=device)
    Theta_t = torch.tensor(Theta_np.reshape(-1, 1), dtype=torch.float32, device=device)

    Vr_PP_t = pinn_predict(model_Vr_PP, p_Vr_PP, R_t, Theta_t).cpu().numpy().flatten()
    Vt_PP_t = pinn_predict(model_Vt_PP, p_Vt_PP, R_t, Theta_t).cpu().numpy().flatten()
    Vr_Os_t = pinn_predict(model_Vr_Oseen, p_Vr_Oseen, R_t, Theta_t).cpu().numpy().flatten()
    Vt_Os_t = pinn_predict(model_Vt_Oseen, p_Vt_Oseen, R_t, Theta_t).cpu().numpy().flatten()

    pp_mask = R_np <= 2
    blend_mask = (R_np > 2) & (R_np < 5)
    oseen_mask = R_np >= 5

    vr = np.zeros_like(R_np)
    vt = np.zeros_like(R_np)

    vr[pp_mask] = Vr_PP_t[pp_mask]
    vt[pp_mask] = Vt_PP_t[pp_mask]

    vr[oseen_mask] = Vr_Os_t[oseen_mask]
    vt[oseen_mask] = Vt_Os_t[oseen_mask]

    R_b = R_np[blend_mask]
    vr[blend_mask] = ((5 - R_b) * Vr_PP_t[blend_mask] + (R_b - 2) * Vr_Os_t[blend_mask]) / 3
    vt[blend_mask] = ((5 - R_b) * Vt_PP_t[blend_mask] + (R_b - 2) * Vt_Os_t[blend_mask]) / 3

    cos_th = np.cos(Theta_np)
    sin_th = np.sin(Theta_np)
    U_valid = vr * cos_th - vt * sin_th
    V_valid = vr * sin_th + vt * cos_th

    source_speed = np.linalg.norm(p_vel)
    U_valid = U_valid * source_speed
    V_valid = V_valid * source_speed

    U_flat = np.zeros(r.shape)
    V_flat = np.zeros(r.shape)
    U_flat[valid] = np.where(np.isfinite(U_valid), U_valid, 0.0)
    V_flat[valid] = np.where(np.isfinite(V_valid), V_valid, 0.0)

    return U_flat, V_flat


# ============ 动画（与 WAKE_PINN_test.py 完全相同的视场/几何） ============

dt = 0.02
steps = 300

particle_pos0 = np.array([0.008, 0.0])
particle_vel = np.array([-1.0, 0.0])
U_REF = float(np.linalg.norm(particle_vel))


def build_polar_ring_points(radius, n_rings, center=(0.0, 0.0)):
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


grid_size = 18
DOMAIN_RADIUS = 5e-3
X, Y = build_polar_ring_points(DOMAIN_RADIUS, grid_size)
_mask_outside = np.hypot(X, Y) > _A_P
X = X[_mask_outside]
Y = Y[_mask_outside]
DOMAIN_HALF_R = DOMAIN_RADIUS / _A_P
X_R = X / _A_P
Y_R = Y / _A_P

fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0d1117")
ax.set_facecolor("#0d1117")

U0 = np.zeros_like(X)
V0 = np.zeros_like(Y)
speed0 = np.zeros_like(X)

cmap = plt.cm.plasma
norm_color = mcolors.Normalize(vmin=0, vmax=max(U_REF * 1e-6, 1e-9))
ARROW_SCALE = 1.75e-4
ARROW_SCALE_R = ARROW_SCALE / _A_P

quiver = ax.quiver(
    X_R, Y_R, U0, V0, speed0,
    cmap=cmap, norm=norm_color,
    scale=1,
    scale_units="xy",
    width=0.0018,
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

particle_plot, = ax.plot([], [], "o",
                          color="#00e5ff", markersize=9,
                          markeredgecolor="white", markeredgewidth=0.8,
                          zorder=5)

ax.set_xlim(-DOMAIN_HALF_R, DOMAIN_HALF_R)
ax.set_ylim(-DOMAIN_HALF_R, DOMAIN_HALF_R)
ax.set_aspect("equal")
ax.set_xlabel(r"$x\,/\,a$", color="white", fontsize=11)
ax.set_ylabel(r"$y\,/\,a$", color="white", fontsize=11)
ax.set_title(
    f"Particle Wake (DDPINN, Re_closure={WAKE_CLOSURE_RE_DEFAULT:g}, |U|={U_REF:g} m/s)",
    color="white",
    fontsize=13,
    pad=12,
)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")

time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                    color="white", fontsize=10, va="top")


def update(frame):
    particle_pos = particle_pos0 + particle_vel * dt * frame

    U, V = compute_flow_field_ddpinn(particle_pos, particle_vel)
    speed = np.sqrt(U ** 2 + V ** 2)

    eps = 1e-12
    U_norm = U / (speed + eps) * ARROW_SCALE_R
    V_norm = V / (speed + eps) * ARROW_SCALE_R

    quiver.set_UVC(U_norm, V_norm, speed)
    s_max = float(np.percentile(speed, 99.0)) if np.any(speed > 0) else max(U_REF * 1e-9, 1e-12)
    norm_color.vmax = max(s_max, max(U_REF * 1e-9, 1e-12))

    particle_plot.set_data([0.0], [0.0])
    time_text.set_text(f"t = {frame * dt:.2f} s\nframe: particle-centered")

    return quiver, particle_plot, time_text


update(0)
plt.tight_layout()
fig.savefig(STATIC_OUT, dpi=220, facecolor=fig.get_facecolor())
print(f"Saved static: {os.path.basename(STATIC_OUT)}")

if __name__ == "__main__" and os.environ.get("MPLBACKEND", "").lower() != "agg":
    anim = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=50,
        blit=False,
        repeat=True,
    )
    plt.show()
