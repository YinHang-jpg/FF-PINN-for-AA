"""DDPINN 统合模型：把 4 个子 PINN 按 DEM_wake 公式组合成一个端到端可微分网络。

输入   : 相对粒子中心的笛卡尔位移 (x_m, y_m) [m] 与粒子速度模 |U| [m/s]。
输出   : 该点处的笛卡尔速度分量 (u, v) [m/s]，与 COMSOL 数据维度量纲一致。

组合方式（与 wake_effect/DEM_wake/DEM_wake.py 中 acoustic_wake_velocity 一致）：
  1) 由 (x_m, y_m) 计算 r、R = r/a、Theta = atan2(y, x)；
  2) 4 个子网络分别输出 vr_PP, vt_PP, vr_Oseen, vt_Oseen（无量纲）；
  3) 分段混合权重: w_os = clamp((R-2)/3, 0, 1)，w_pp = 1 - w_os
        - R ≤ 2 → 纯 PP, R ≥ 5 → 纯 Oseen, 2 < R < 5 → 线性混合
  4) 极坐标合成笛卡尔: u = vr*cos(Θ) - vt*sin(Θ), v = vr*sin(Θ) + vt*cos(Θ)；
  5) 乘以 |U| 恢复 m/s 量纲。
"""
from __future__ import annotations

import torch
import torch.nn as nn


# 子网络结构必须与 wake_effect/PINN_wake/WAKE_PINN_*.py 完全一致，
# 否则 .pth 加载时会报形状/键名错误。
class WakeNetVrPP(nn.Module):
    def __init__(self, n_harmonics: int = 8) -> None:
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled: torch.Tensor, Theta: torch.Tensor) -> torch.Tensor:
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVtPP(nn.Module):
    def __init__(self, n_harmonics: int = 8) -> None:
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled: torch.Tensor, Theta: torch.Tensor) -> torch.Tensor:
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVrOseen(nn.Module):
    def __init__(self, n_harmonics: int = 16) -> None:
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled: torch.Tensor, Theta: torch.Tensor) -> torch.Tensor:
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


class WakeNetVtOseen(nn.Module):
    def __init__(self, n_harmonics: int = 16) -> None:
        super().__init__()
        self.register_buffer(
            "harmonics", torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        )
        d = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(d, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, R_scaled: torch.Tensor, Theta: torch.Tensor) -> torch.Tensor:
        p = self.harmonics * Theta
        x = torch.cat([R_scaled, torch.sin(p), torch.cos(p)], dim=1)
        return self.layers(x)


# 子网络分组：name → (类, 默认 norm.json 文件名, 默认 .pth 文件名)
_NET_SPECS = (
    ("vr_pp",    WakeNetVrPP,    "wake_Vr_PP_norm.json",    "wake_Vr_PP.pth"),
    ("vt_pp",    WakeNetVtPP,    "wake_Vt_PP_norm.json",    "wake_Vt_PP.pth"),
    ("vr_oseen", WakeNetVrOseen, "wake_Vr_Oseen_norm.json", "wake_Vr_Oseen.pth"),
    ("vt_oseen", WakeNetVtOseen, "wake_Vt_Oseen_norm.json", "wake_Vt_Oseen.pth"),
)


def net_specs() -> tuple:
    """返回 _NET_SPECS（供外部脚本枚举）。"""
    return _NET_SPECS


class CombinedWakeModel(nn.Module):
    """端到端的 (x, y) → (u, v) 模型；包含 4 个可训练子 PINN + DEM_wake 组合公式。

    每个子网络的归一化参数 (R_min, R_max, force_mu, force_sigma) 注册为 buffer，
    在微调过程中保持不变（与 PINN_wake 训练时保存的 norm.json 一一对应）。
    """

    def __init__(self, char_length_m: float) -> None:
        super().__init__()
        if char_length_m <= 0.0:
            raise ValueError(f"char_length_m 必须为正: {char_length_m}")
        self.char_length_m = float(char_length_m)

        self.net_vr_pp = WakeNetVrPP(n_harmonics=8)
        self.net_vt_pp = WakeNetVtPP(n_harmonics=8)
        self.net_vr_oseen = WakeNetVrOseen(n_harmonics=16)
        self.net_vt_oseen = WakeNetVtOseen(n_harmonics=16)

        for name, *_ in _NET_SPECS:
            self.register_buffer(f"{name}_R_min", torch.tensor(0.0))
            self.register_buffer(f"{name}_R_max", torch.tensor(1.0))
            self.register_buffer(f"{name}_mu", torch.tensor(0.0))
            self.register_buffer(f"{name}_sigma", torch.tensor(1.0))

    # ---------- 归一化参数 ----------
    def set_norm(self, name: str, R_min: float, R_max: float, mu: float, sigma: float) -> None:
        getattr(self, f"{name}_R_min").fill_(float(R_min))
        getattr(self, f"{name}_R_max").fill_(max(float(R_max), float(R_min) + 1e-12))
        getattr(self, f"{name}_mu").fill_(float(mu))
        getattr(self, f"{name}_sigma").fill_(max(float(sigma), 1e-30))

    def get_norm(self, name: str) -> dict:
        return {
            "R_min": float(getattr(self, f"{name}_R_min").item()),
            "R_max": float(getattr(self, f"{name}_R_max").item()),
            "force_mu": float(getattr(self, f"{name}_mu").item()),
            "force_sigma": float(getattr(self, f"{name}_sigma").item()),
        }

    def get_subnet(self, name: str) -> nn.Module:
        return getattr(self, f"net_{name}")

    # ---------- 子网络调用 ----------
    def _predict_one(self, name: str, R: torch.Tensor, Theta: torch.Tensor) -> torch.Tensor:
        net = self.get_subnet(name)
        R_min = getattr(self, f"{name}_R_min")
        R_max = getattr(self, f"{name}_R_max")
        mu = getattr(self, f"{name}_mu")
        sigma = getattr(self, f"{name}_sigma")
        R_scaled = torch.clamp((R - R_min) / (R_max - R_min + 1e-30), 0.0, 1.0)
        out_norm = net(R_scaled, Theta)
        return out_norm * sigma + mu  # 无量纲物理量

    # ---------- 端到端正向 ----------
    def forward_xy(
        self,
        x_m: torch.Tensor,
        y_m: torch.Tensor,
        source_speed_m_s: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_m, y_m)[m] + |U|[m/s] → (u, v)[m/s]。"""
        eps = 1e-30
        r = torch.sqrt(x_m * x_m + y_m * y_m + eps)
        R = r / self.char_length_m
        Theta = torch.atan2(y_m, x_m)

        vr_pp = self._predict_one("vr_pp", R, Theta)
        vt_pp = self._predict_one("vt_pp", R, Theta)
        vr_os = self._predict_one("vr_oseen", R, Theta)
        vt_os = self._predict_one("vt_oseen", R, Theta)

        # 平滑混合权重（与 DEM_wake.acoustic_wake_velocity 完全一致）：
        #   R ≤ 2 → w_os=0；R ≥ 5 → w_os=1；2 < R < 5 → 线性。
        w_os = torch.clamp((R - 2.0) / 3.0, 0.0, 1.0)
        w_pp = 1.0 - w_os
        vr = w_pp * vr_pp + w_os * vr_os
        vt = w_pp * vt_pp + w_os * vt_os

        cos_t = torch.cos(Theta)
        sin_t = torch.sin(Theta)
        u_dimless = vr * cos_t - vt * sin_t
        v_dimless = vr * sin_t + vt * cos_t

        s = source_speed_m_s
        if not torch.is_tensor(s):
            s = torch.tensor(float(s), dtype=u_dimless.dtype, device=u_dimless.device)
        u_m_s = u_dimless * s
        v_m_s = v_dimless * s
        return u_m_s, v_m_s

    # ---------- 子网络在极坐标上的预测（用于物理正则项） ----------
    def predict_subnet(
        self, name: str, R: torch.Tensor, Theta: torch.Tensor
    ) -> torch.Tensor:
        """返回子网络的去归一化输出（即无量纲极坐标速度分量）。"""
        return self._predict_one(name, R, Theta)

    def predict_subnet_normalized(
        self, name: str, R: torch.Tensor, Theta: torch.Tensor
    ) -> torch.Tensor:
        """返回子网络的归一化输出（直接来自最后一层），用于在归一化空间计算物理 MSE。"""
        net = self.get_subnet(name)
        R_min = getattr(self, f"{name}_R_min")
        R_max = getattr(self, f"{name}_R_max")
        R_scaled = torch.clamp((R - R_min) / (R_max - R_min + 1e-30), 0.0, 1.0)
        return net(R_scaled, Theta)
