"""
Wake PINN 批量推理（与 WAKE_PINN_test / DEM_wake 约定一致）。

独立成模块的原因：避免在 comsol_vs_DEM 等处重复网络定义与分段混合逻辑；
勿改为直接 import WAKE_PINN_test（其会在 import 时加载权重、打印并依赖 matplotlib）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# DEM_wake：无量纲 R、强度缩放
_PINN_WAKE_DIR = Path(__file__).resolve().parent
_WAKE_EFFECT_DIR = _PINN_WAKE_DIR.parent
_DEM_WAKE_DIR = _WAKE_EFFECT_DIR / "DEM_wake"
_PINN_FALLBACK_DIR = _WAKE_EFFECT_DIR / "PINN"

if str(_DEM_WAKE_DIR) not in sys.path:
    sys.path.insert(0, str(_DEM_WAKE_DIR))
from DEM_wake import WAKE_STRENGTH_MULTIPLIER  # noqa: E402


def default_pinn_weight_dir() -> Path:
    """优先 PINN_wake/PINN 下的 .pth，否则退回 wake_effect/PINN。"""
    d1 = _PINN_WAKE_DIR / "PINN"
    if (d1 / "wake_Vr_PP.pth").is_file():
        return d1
    return _PINN_FALLBACK_DIR


class WakeNetVrPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled, Theta):
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVtPP(nn.Module):
    def __init__(self, n_harmonics=8):
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled, Theta):
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVrOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled, Theta):
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVtOseen(nn.Module):
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled, Theta):
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


def _load_one(
    base: Path,
    model_class: type[nn.Module],
    pth_name: str,
    json_name: str,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    with open(base / json_name, "r", encoding="utf-8") as f:
        params = json.load(f)
    m = model_class()
    m.load_state_dict(
        torch.load(base / pth_name, map_location=device, weights_only=True)
    )
    m.to(device).eval()
    return m, params


def load_wake_pinn_bundle(
    device: torch.device | None = None,
    weight_dir: Path | None = None,
) -> dict:
    """加载 4 个子网络；缺少 .pth 时抛 FileNotFoundError。"""
    if device is None:
        device = torch.device("cpu")
    base = Path(weight_dir) if weight_dir is not None else default_pinn_weight_dir()
    specs = [
        ("Vr_PP", WakeNetVrPP, "wake_Vr_PP.pth", "wake_Vr_PP_norm.json"),
        ("Vt_PP", WakeNetVtPP, "wake_Vt_PP.pth", "wake_Vt_PP_norm.json"),
        ("Vr_Oseen", WakeNetVrOseen, "wake_Vr_Oseen.pth", "wake_Vr_Oseen_norm.json"),
        ("Vt_Oseen", WakeNetVtOseen, "wake_Vt_Oseen.pth", "wake_Vt_Oseen_norm.json"),
    ]
    out: dict = {"weight_dir": base}
    for key, cls, pth, js in specs:
        p = base / pth
        if not p.is_file():
            raise FileNotFoundError(f"缺少 PINN 权重: {p}")
        out[key] = _load_one(base, cls, pth, js, device)
    return out


def pinn_predict(
    model: nn.Module, params: dict, R_tensor: torch.Tensor, Theta_tensor: torch.Tensor
) -> torch.Tensor:
    R_scaled = (R_tensor - params["R_min"]) / (params["R_max"] - params["R_min"])
    R_scaled = torch.clamp(R_scaled, 0.0, 1.0)
    with torch.no_grad():
        pred_norm = model(R_scaled, Theta_tensor)
    return pred_norm * params["force_sigma"] + params["force_mu"]


def pinn_velocities_m_s(
    x_m: np.ndarray,
    y_m: np.ndarray,
    particle_vel: np.ndarray,
    char_length_m: float,
    bundle: dict,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    粒子系：采样点 (x_m,y_m) 为相对粒子中心的位移 [m]；
    返回与 x_m 同形状的 u,v [m/s]。
    """
    if device is None:
        device = torch.device("cpu")

    m_vr_pp, p_vr_pp = bundle["Vr_PP"]
    m_vt_pp, p_vt_pp = bundle["Vt_PP"]
    m_vr_os, p_vr_os = bundle["Vr_Oseen"]
    m_vt_os, p_vt_os = bundle["Vt_Oseen"]

    r = np.hypot(x_m, y_m)
    valid = r > 1e-12
    R_np = r[valid] / char_length_m
    Theta_np = np.arctan2(y_m[valid], x_m[valid])

    R_t = torch.tensor(R_np.reshape(-1, 1), dtype=torch.float32, device=device)
    Th_t = torch.tensor(Theta_np.reshape(-1, 1), dtype=torch.float32, device=device)

    Vr_PP_t = pinn_predict(m_vr_pp, p_vr_pp, R_t, Th_t).cpu().numpy().flatten()
    Vt_PP_t = pinn_predict(m_vt_pp, p_vt_pp, R_t, Th_t).cpu().numpy().flatten()
    Vr_Os_t = pinn_predict(m_vr_os, p_vr_os, R_t, Th_t).cpu().numpy().flatten()
    Vt_Os_t = pinn_predict(m_vt_os, p_vt_os, R_t, Th_t).cpu().numpy().flatten()

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

    src_sp = float(np.linalg.norm(particle_vel))
    scale = float(WAKE_STRENGTH_MULTIPLIER) * src_sp
    U_valid = U_valid * scale
    V_valid = V_valid * scale

    U_flat = np.zeros_like(x_m, dtype=float)
    V_flat = np.zeros_like(y_m, dtype=float)
    U_flat[valid] = np.where(np.isfinite(U_valid), U_valid, 0.0)
    V_flat[valid] = np.where(np.isfinite(V_valid), V_valid, 0.0)
    return U_flat, V_flat
