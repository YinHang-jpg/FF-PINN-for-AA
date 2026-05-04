"""
无量纲 Oseen / Proudman–Pearson 的 Vr、Vt（Torch），与 DEM_wake.acoustic_wake_velocity 逐行一致。

约定：
  - 无量纲半径 R = r / a（a = CHAR_LENGTH_M）；
  - 极角 Theta = arctan2(y, x)，其中 (x,y) 为 target − source（与 DEM_wake 一致）；
  - 笛卡尔合成：vx = vr*cos(Theta) - vt*sin(Theta)，vy = vr*sin(Theta) + vt*cos(Theta)。
"""
from __future__ import annotations

import torch


def dimless_vr_oseen(R: torch.Tensor, Theta: torch.Tensor, Re) -> torch.Tensor:
    """DEM_wake: Oseen_r_term1/2/3 → Vr_Oseen。"""
    cos_t = torch.cos(Theta)
    exp_arg = torch.clamp((-R * Re / 4.0) * (1.0 - cos_t), -500.0, 500.0)
    exp_term = torch.exp(exp_arg)
    R2 = R * R
    term1 = cos_t / (2.0 * R)
    term2 = (3.0 * R * (1.0 + cos_t) / 4.0) * exp_term
    term3 = 3.0 * (1.0 - exp_term) / Re
    return (term1 - term2 + term3) / R2


def dimless_vt_oseen(R: torch.Tensor, Theta: torch.Tensor, Re) -> torch.Tensor:
    """DEM_wake: Vt_Oseen = sin_t * (Oseen_t_term1 + Oseen_t_term2) / (4*R)。"""
    cos_t = torch.cos(Theta)
    sin_t = torch.sin(Theta)
    exp_arg = torch.clamp((-R * Re / 4.0) * (1.0 - cos_t), -500.0, 500.0)
    exp_term = torch.exp(exp_arg)
    R2 = R * R
    return sin_t * (1.0 / R2 + 3.0 * exp_term) / (4.0 * R)


def dimless_vr_pp(R: torch.Tensor, Theta: torch.Tensor, Re) -> torch.Tensor:
    """DEM_wake: PP_r_term1/2 → Vr_PP。"""
    cos_t = torch.cos(Theta)
    cos_2t = torch.cos(2.0 * Theta)
    R2 = R * R
    R4 = R2 * R2
    PP_r_term1 = (
        4.0 * R * (16.0 + 3.0 * Re + 3.0 * R2 * (-16.0 + (2.0 * R - 3.0) * Re)) * cos_t
    )
    PP_r_term2 = (
        3.0 * ((R - 1.0) ** 2) * (1.0 + R + 2.0 * R2) * (1.0 + 3.0 * cos_2t) * Re
    )
    return (PP_r_term1 - PP_r_term2) / (128.0 * R4)


def dimless_vt_pp(R: torch.Tensor, Theta: torch.Tensor, Re) -> torch.Tensor:
    """DEM_wake: PP_t_term1/2 → Vt_PP。"""
    cos_t = torch.cos(Theta)
    sin_t = torch.sin(Theta)
    R2 = R * R
    R4 = R2 * R2
    PP_t_term1 = R * (16.0 + 3.0 * Re + 3.0 * R2 * (16.0 + (3.0 - 4.0 * R) * Re))
    PP_t_term2 = 3.0 * (-2.0 + R - 3.0 * R ** 3 + 4.0 * R4) * cos_t * Re
    return sin_t * (PP_t_term1 + PP_t_term2) / (64.0 * R4)


def blend_vr_vt(
    R: torch.Tensor,
    vr_pp: torch.Tensor,
    vt_pp: torch.Tensor,
    vr_os: torch.Tensor,
    vt_os: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    与 DEM_wake 分段一致：
      R <= 2  → 纯 PP；2 < R < 5 → 线性混合；R >= 5 → 纯 Oseen。
    """
    pp = R <= 2.0
    os_m = R >= 5.0
    w = (R - 2.0) / 3.0
    vr = torch.where(pp, vr_pp, torch.where(os_m, vr_os, (1.0 - w) * vr_pp + w * vr_os))
    vt = torch.where(pp, vt_pp, torch.where(os_m, vt_os, (1.0 - w) * vt_pp + w * vt_os))
    return vr, vt


def polar_to_cartesian(vr: torch.Tensor, vt: torch.Tensor, Theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DEM_wake: vx = vr*cos(Theta) - vt*sin(Theta)，vy = vr*sin(Theta) + vt*cos(Theta)。"""
    c = torch.cos(Theta)
    s = torch.sin(Theta)
    return vr * c - vt * s, vr * s + vt * c
