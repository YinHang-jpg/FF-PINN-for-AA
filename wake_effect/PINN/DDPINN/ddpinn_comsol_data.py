"""DDPINN：COMSOL 高保真度数据加载与极坐标投影。

注意事项（与用户提醒一一对应）：
  1. 量纲：COMSOL_vx.txt / COMSOL_vy.txt 直接给出 (vx, vy) 单位 m/s；
     PINN 子网络在 |U|=1 m/s 算例下的“无量纲”输出与 m/s 数值相同（除以 |U|）。
     本模块统一返回除以 ``u_ref_m_s`` 后的无量纲分量。
  2. 方向：COMSOL 数据是笛卡尔系；PINN 子网络以极坐标 (R, Theta) 拆分 Vr / Vt。
     这里通过 vr =  vx*cos(Θ) + vy*sin(Θ),  vt = -vx*sin(Θ) + vy*cos(Θ)
     把笛卡尔速度投影到极坐标基。
  3. 范围：仅保留 r ∈ (a, r_max] 的节点（a 为粒子半径，r_max 为用户考察上限）。

返回的 ``R`` 已是 r / char_length_m（无量纲），与各 PINN 子网络的输入约定一致。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_DDPINN_DIR = Path(__file__).resolve().parent
_PINN_DIR = _DDPINN_DIR.parent
_WAKE_EFFECT_DIR = _PINN_DIR.parent
_COMSOL_DIR = _WAKE_EFFECT_DIR / "COMSOL_wake"

DEFAULT_VX_PATH = _COMSOL_DIR / "COMSOL_vx.txt"
DEFAULT_VY_PATH = _COMSOL_DIR / "COMSOL_vy.txt"


def _load_comsol_table(path: Path) -> np.ndarray:
    """读取 COMSOL 文本导出（注释行以 % 开头），返回 (N, ncols) 的 float 数组。"""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            rows.append([float(x) for x in s.split()])
    if not rows:
        raise RuntimeError(f"COMSOL 文件无有效数据行: {path}")
    return np.asarray(rows, dtype=float)


def load_comsol_polar(
    char_length_m: float,
    u_ref_m_s: float,
    r_max_m: float,
    *,
    vx_path: Path | str = DEFAULT_VX_PATH,
    vy_path: Path | str = DEFAULT_VY_PATH,
) -> dict:
    """加载 COMSOL，过滤至 r ∈ (a, r_max]，并把 (vx, vy) 投影到 (Vr, Vt)（无量纲）。

    Parameters
    ----------
    char_length_m : float
        粒子半径 a [m]，用于把 r 无量纲化为 R = r / a。
    u_ref_m_s : float
        参考来流速度 |U| [m/s]。COMSOL 数据除以 |U| 得到无量纲速度，
        与 PINN 子网络输出（无量纲）保持一致。
    r_max_m : float
        用户考察的最大半径 [m]（典型 5e-3 = 5 mm）。
    vx_path, vy_path : Path | str
        COMSOL 导出文件路径；默认指向 wake_effect/COMSOL_wake 下两个文件。

    Returns
    -------
    dict
        keys: R, Theta, Vr, Vt（均 float32 numpy 数组，均无量纲），以及
        x_m, y_m, vx_m_s, vy_m_s（保留原始量纲，便于诊断/绘图）。
    """
    if char_length_m <= 0.0:
        raise ValueError(f"char_length_m 必须为正: {char_length_m}")
    if u_ref_m_s <= 0.0:
        raise ValueError(f"u_ref_m_s 必须为正: {u_ref_m_s}")
    if r_max_m <= char_length_m:
        raise ValueError(
            f"r_max_m={r_max_m} 必须大于粒子半径 a={char_length_m}"
        )

    vx_tab = _load_comsol_table(Path(vx_path))
    vy_tab = _load_comsol_table(Path(vy_path))
    if vx_tab.shape != vy_tab.shape:
        raise RuntimeError(
            f"COMSOL vx/vy 形状不一致: {vx_tab.shape} vs {vy_tab.shape}"
        )
    if not np.allclose(vx_tab[:, :2], vy_tab[:, :2], atol=1e-12):
        raise RuntimeError("COMSOL vx/vy 坐标列不匹配（前两列应完全一致）")

    x = vx_tab[:, 0].astype(np.float64)
    y = vx_tab[:, 1].astype(np.float64)
    vx = vx_tab[:, -1].astype(np.float64)
    vy = vy_tab[:, -1].astype(np.float64)

    r = np.hypot(x, y)
    mask = (
        (r > char_length_m)
        & (r <= r_max_m)
        & np.isfinite(vx)
        & np.isfinite(vy)
    )
    if not np.any(mask):
        raise RuntimeError(
            f"COMSOL 数据在 r ∈ ({char_length_m}, {r_max_m}] 内无有效节点"
        )

    x = x[mask]
    y = y[mask]
    vx = vx[mask]
    vy = vy[mask]
    r = r[mask]

    Theta = np.arctan2(y, x)
    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    Vr_m_s = vx * cos_t + vy * sin_t
    Vt_m_s = -vx * sin_t + vy * cos_t

    R = r / char_length_m
    Vr_dimless = Vr_m_s / u_ref_m_s
    Vt_dimless = Vt_m_s / u_ref_m_s

    return {
        "R": R.astype(np.float32),
        "Theta": Theta.astype(np.float32),
        "Vr": Vr_dimless.astype(np.float32),
        "Vt": Vt_dimless.astype(np.float32),
        "x_m": x.astype(np.float32),
        "y_m": y.astype(np.float32),
        "vx_m_s": vx.astype(np.float32),
        "vy_m_s": vy.astype(np.float32),
    }


def subset_R(data: dict, R_min: float, R_max: float) -> dict:
    """按无量纲 R 区间筛选返回字典中的所有数组。"""
    R = data["R"]
    sel = (R >= R_min) & (R <= R_max)
    out = {k: v[sel] for k, v in data.items()}
    return out
