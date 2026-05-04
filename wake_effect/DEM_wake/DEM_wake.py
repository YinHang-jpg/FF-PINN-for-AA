import numpy as np

"""粒子系尾流（Oseen/PP 分段 + 笛卡尔合成）。

**粒子雷诺数（基于直径）**：

    Re_D = particle_reynolds_diameter(|U|) = |U| D / ν

其中 ``D = _D_P`` 为粒子直径，``ν = μ/ρ`` 为运动粘度。

无量纲半径仍取 ``R = r / a``（``a = _A_P`` 为半径），与 Oseen/PP 分段一致。

``acoustic_wake_velocity(..., Re=...)`` 的形参 ``Re`` 出现在 ``R*Re/4`` 等闭式项中；
``WAKE_CLOSURE_RE_DEFAULT`` 在 **|U|=1 m/s** 时与 ``Re_D`` 对齐为 **1**
（ρ=1 kg/m³，μ=1e-3 Pa·s ⇒ ν=1e-3 m²/s，``|U| D / ν = 1``）。
"""

# 特征长度 [m]：与粒子半径一致，使无量纲半径 R = r / a（与 Oseen/PP 尾流公式一致）
# 对解析式括号内项不变，仅在恢复 m/s 后做全局强度标定（例如与 2D CFD 幅值对齐时）。
# 与 PINN 训练数据对比时请保持为 1.0，以免与网络归一化不一致。
WAKE_STRENGTH_MULTIPLIER = 1.0

# 实际粒子物理参数 (用于量纲分析诊断)
# 二维圆形粒子：半径 500 μm，直径 1 mm
_D_P = 1e-3        # 粒子直径 [m]
_A_P = _D_P / 2    # 粒子半径 [m] — Oseen/PP 公式中正确的特征长度
CHAR_LENGTH_M = _A_P

# 等效流体：ρ = 1 kg/m³，μ = 1e-3 Pa·s ⇒ ν = 1e-3 m²/s
# 粒子雷诺数按直径：Re_D = |U| D / ν；|U|=1、D=_D_P=1e-3 时 Re_D = 1
# （符号名 _RHO_WATER/_MU_WATER 保留便于旧代码导入）
_RHO_WATER = 1.0
_MU_WATER = 1e-3
_NU_WATER = _MU_WATER / _RHO_WATER


def kinematic_reynolds(speed: float, characteristic_length_m: float, nu: float) -> float:
    """Re = |U| L / ν（L 为所选特征长度 [m]，如直径 D 或半径 a；ν 为运动粘度 [m²/s]）。"""
    return float(abs(speed)) * float(characteristic_length_m) / max(float(nu), 1e-30)


def particle_reynolds_diameter(speed: float) -> float:
    """粒子雷诺数（基于直径）Re_D = |U| D / ν，D = _D_P。"""
    return kinematic_reynolds(speed, _D_P, _NU_WATER)


# Oseen/PP 闭式（及 PINN）中 ``R*Re/4`` 等项的默认参量；参考 |U|=1 m/s 时与 Re_particle 一致。
WAKE_CLOSURE_RE_DEFAULT = 1.0


def acoustic_wake_velocity(source_pos, target_pos, source_vel, Re, return_cartesian=True):
    """粒子系尾流速度（无量纲 Oseen/PP + 有量纲缩放）。

    参数 ``Re``：无量纲闭式中的参量（与文献/PINN 一致时通常取
    ``WAKE_CLOSURE_RE_DEFAULT``）。粒子雷诺数（直径定义）请用
    ``particle_reynolds_diameter(|U|)`` 或 ``kinematic_reynolds(|U|, _D_P, ν)``。
    """
    source_pos = np.asarray(source_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    source_vel = np.asarray(source_vel, dtype=np.float64)
    source_speed = np.linalg.norm(source_vel)

    rel_vec = target_pos - source_pos
    x = rel_vec[0]
    y = rel_vec[1]

    r = np.hypot(x, y)
    if r < 1e-12:
        return (0.0, 0.0)

    R = r / CHAR_LENGTH_M

    Theta = np.arctan2(y, x)

    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    cos_2t = np.cos(2 * Theta)

    # Oseen terms
    Oseen_r_term1 = cos_t / (2 * R)
    Oseen_r_term2 = (3 * R * (1 + cos_t) / 4) * np.exp((-R * Re / 4) * (1 - cos_t))
    Oseen_r_term3 = 3 * (1 - np.exp((-R * Re / 4) * (1 - cos_t))) / Re
    Vr_Oseen = (Oseen_r_term1 - Oseen_r_term2 + Oseen_r_term3) / (R ** 2)

    Oseen_t_term1 = 1 / (R ** 2)
    Oseen_t_term2 = 3 * np.exp((-R * Re / 4) * (1 - cos_t))
    Vt_Oseen = sin_t * (Oseen_t_term1 + Oseen_t_term2) / (4 * R)

    # PP terms
    PP_r_term1 = 4 * R * (16 + 3 * Re + 3 * R ** 2 * (-16 + (2 * R - 3) * Re)) * cos_t
    PP_r_term2 = 3 * ((R - 1) ** 2) * (1 + R + 2 * R ** 2) * (1 + 3 * cos_2t) * Re
    Vr_PP = (PP_r_term1 - PP_r_term2) / (128 * R ** 4)

    PP_t_term1 = R * (16 + 3 * Re + 3 * R ** 2 * (16 + (3 - 4 * R) * Re))
    PP_t_term2 = 3 * (-2 + R - 3 * R ** 3 + 4 * R ** 4) * cos_t * Re
    Vt_PP = sin_t * (PP_t_term1 + PP_t_term2) / (64 * R ** 4)

    # 分段（r 为相对粒子中心的距离，R = r/a；图中 2、5 均为 a 的倍数）：
    #   r <= 2a：纯 PP；2a < r < 5a：线性混合；r >= 5a：纯 Oseen
    if R <= 2:
        vr = Vr_PP
        vt = Vt_PP
    elif R < 5:
        vr = ((5 - R) * Vr_PP + (R - 2) * Vr_Oseen) / 3
        vt = ((5 - R) * Vt_PP + (R - 2) * Vt_Oseen) / 3
    else:
        vr = Vr_Oseen
        vt = Vt_Oseen

    if not return_cartesian:
        # Vr/Vt 由公式给出为无量纲速度，这里乘以源粒子速度恢复到 m/s
        s = float(WAKE_STRENGTH_MULTIPLIER) * source_speed
        return (vr * s, vt * s)

    # 公式输出为无量纲速度，乘以源粒子速度得到有量纲速度 [m/s]
    # 标准极坐标合成 v = vr e_r + vt e_θ（Θ = arctan2(y,x)）；勿再对 vx 做半平面 abs 修正，
    # 否则与 COMSOL 等 CFD 导出的物理速度场不一致（对比图在 2<r/a<5 尤为明显）。
    s = float(WAKE_STRENGTH_MULTIPLIER) * source_speed
    vx_global = (vr * np.cos(Theta) - vt * np.sin(Theta)) * s
    vy_global = (vr * np.sin(Theta) + vt * np.cos(Theta)) * s

    if not np.isfinite(vx_global) or not np.isfinite(vy_global):
        return (0.0, 0.0)

    return (vx_global, vy_global)


def _compute_Vr_Oseen_raw(R, Theta, Re):
    """纯公式计算 Oseen 径向无量纲速度 (不含任何方向修正)"""
    cos_t = np.cos(Theta)
    exp_arg = (-R * Re / 4.0) * (1.0 - cos_t)
    exp_term = np.exp(np.clip(exp_arg, -500, 500))
    term1 = cos_t / (2.0 * R)
    term2 = (3.0 * R * (1.0 + cos_t) / 4.0) * exp_term
    term3 = 3.0 * (1.0 - exp_term) / Re
    return (term1 - term2 + term3) / (R ** 2)


def _compute_Vr_PP_raw(R, Theta, Re):
    """纯公式计算 PP 径向无量纲速度"""
    cos_t = np.cos(Theta)
    cos_2t = np.cos(2.0 * Theta)
    term1 = 4.0 * R * (16.0 + 3.0 * Re + 3.0 * R**2 * (-16.0 + (2.0*R - 3.0)*Re)) * cos_t
    term2 = 3.0 * ((R - 1.0)**2) * (1.0 + R + 2.0*R**2) * (1.0 + 3.0*cos_2t) * Re
    return (term1 - term2) / (128.0 * R**4)


def _compute_Vt_Oseen_raw(R, Theta, Re):
    """纯公式 Oseen 切向无量纲速度（与 acoustic_wake_velocity 中 Vt_Oseen 一致）"""
    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    exp_arg = (-R * Re / 4.0) * (1.0 - cos_t)
    exp_term = np.exp(np.clip(exp_arg, -500, 500))
    R2 = R ** 2
    return sin_t * (1.0 / R2 + 3.0 * exp_term) / (4.0 * R)


def _compute_Vt_PP_raw(R, Theta, Re):
    """纯公式 PP 切向无量纲速度（与 acoustic_wake_velocity 中 Vt_PP 一致）"""
    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    R2 = R ** 2
    R4 = R2 * R2
    PP_t_term1 = R * (16.0 + 3.0 * Re + 3.0 * R2 * (16.0 + (3.0 - 4.0 * R) * Re))
    PP_t_term2 = 3.0 * (-2.0 + R - 3.0 * R ** 3 + 4.0 * R4) * cos_t * Re
    return sin_t * (PP_t_term1 + PP_t_term2) / (64.0 * R4)


def print_dimensional_analysis():
    """打印量纲分析诊断信息和样例数据"""

    re_closure = WAKE_CLOSURE_RE_DEFAULT
    source_vel_example = np.array([-1.0, 0.0])
    U0 = np.linalg.norm(source_vel_example)
    Re_particle = particle_reynolds_diameter(U0)

    rho_p = 2000.0
    mass_p = rho_p * (4.0/3.0) * np.pi * _A_P**3

    drag_coeff = 3.0 * np.pi * _MU_WATER * _D_P / 1.0817

    print("=" * 78)
    print("  DEM_wake.py  量纲分析 & 样例数据")
    print("=" * 78)

    print("\n[1] 物理参数")
    print(f"    粒子直径  d_p       = {_D_P*1e6:.1f} um = {_D_P:.2e} m")
    print(f"    粒子半径  a         = {_A_P*1e6:.1f} um = {_A_P:.2e} m")
    print(f"    粒子密度  rho_p     = {rho_p:.0f} kg/m^3")
    print(f"    粒子质量  m_p       = {mass_p:.4e} kg")
    print(f"    示例粒子速度  U0    = {U0:.4f} m/s ({U0*1e3:.1f} mm/s)")
    print(f"    流体动力粘度  mu    = {_MU_WATER:.2e} Pa.s  (rho={_RHO_WATER:g} kg/m^3)")
    print(f"    流体运动粘度  nu    = {_NU_WATER:.2e} m^2/s")
    print(f"    Stokes 阻力系数     = {drag_coeff:.4e} N/(m/s)")
    print(f"    粒子雷诺数 Re_D=|U0|D/nu = {Re_particle:.4e}  (D 为直径)")
    print(f"    闭式参量 Re_closure     = {re_closure:g}  (Oseen/PP 与 PINN；参考 |U|=1 时与 Re_D 对齐)")

    print("\n" + "-" * 78)
    print("[2] 量纲分析: 当前实现与公式约定")
    print("-" * 78)

    print(f"""
    Oseen/PP 公式的物理含义:
      无量纲半径  R = r / a      (a = 粒子半径)
      公式输出    Vr, Vt         是无量纲速度 = v_physical / U0

    本文件中的实现:
      CHAR_LENGTH_M = a = {_A_P:.3e} m  =>  R = r / a
      有量纲速度: v = U0 * V_dimless，其中 U0 = |source_vel| = {U0:.4f} m/s
      （与径向参考解对比时，表中的「正确物理速度」取 |U0 * Vr_dimless| 便于对照）""")

    print("-" * 78)
    print("[3] 样例数据: 源粒子在原点, U0 = {:.4f} m/s 向左运动".format(U0))
    print("-" * 78)

    source_pos = np.array([0.0, 0.0])

    distances_mm = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    angles_label = [
        ("theta=0 (下游/尾流方向)", 0.0),
        ("theta=pi/2 (侧向)",      np.pi / 2),
        ("theta=pi (上游)",         np.pi),
    ]

    for label, theta in angles_label:
        print(f"\n  方向: {label}")
        print(f"  {'距离':>10s}  {'R(代码)':>10s}  {'R(正确)':>12s}"
              f"  {'代码输出':>14s}  {'正确物理速度':>14s}  {'放大倍数':>10s}")
        print(f"  {'':->10s}  {'':->10s}  {'':->12s}"
              f"  {'':->14s}  {'':->14s}  {'':->10s}")

        for d_mm in distances_mm:
            r = d_mm * 1e-3
            tx = r * np.cos(theta)
            ty = r * np.sin(theta)
            target_pos = np.array([tx, ty])

            vx_code, vy_code = acoustic_wake_velocity(
                source_pos, target_pos, source_vel_example, re_closure)
            v_code = np.hypot(vx_code, vy_code)

            R_code = r / CHAR_LENGTH_M
            R_correct = r / _A_P

            if R_correct <= 2:
                Vr_dimless = _compute_Vr_PP_raw(R_correct, theta, re_closure)
            elif R_correct < 5:
                vr_pp  = _compute_Vr_PP_raw(R_correct, theta, re_closure)
                vr_os  = _compute_Vr_Oseen_raw(R_correct, theta, re_closure)
                Vr_dimless = ((5 - R_correct)*vr_pp + (R_correct - 2)*vr_os) / 3.0
            else:
                Vr_dimless = _compute_Vr_Oseen_raw(R_correct, theta, re_closure)

            v_correct = abs(U0 * Vr_dimless)

            if v_correct > 1e-30:
                ratio = v_code / v_correct
                ratio_str = f"{ratio:.2e}"
            else:
                ratio_str = "inf"

            print(f"  {d_mm:>8.3f}mm  {R_code:>10.2f}  {R_correct:>12.0f}"
                  f"  {v_code:>14.6e}  {v_correct:>14.6e}  {ratio_str:>10s}")

    print("\n" + "-" * 78)
    print("[4] 对模拟的影响 (仿真中的典型参数)")
    print("-" * 78)

    r_typical = 0.5e-3
    R_code_typ = r_typical / CHAR_LENGTH_M
    target_typ = np.array([r_typical, 0.0])
    vx_typ, _ = acoustic_wake_velocity(
        source_pos, target_typ, source_vel_example, re_closure)
    v_typ = abs(vx_typ)

    R_correct_typ = r_typical / _A_P
    Vr_correct_typ = _compute_Vr_Oseen_raw(R_correct_typ, 0.0, re_closure)
    v_correct_typ = abs(U0 * Vr_correct_typ)

    N_neighbors = 100

    print(f"""
    仿真域示例: 34mm x 34mm, 粒子数: ~4400
    平均粒子间距: ~0.5 mm

    在 r = 0.5 mm 处 (尾流方向 theta=0):
      acoustic_wake_velocity 输出 |vx|:   {v_typ:.6e} m/s
      径向参考 |U0*Vr_Oseen|:            {v_correct_typ:.6e} m/s

    粒子自身速度 U0:     {U0:.4e} m/s
    尾流/粒子速度比:     {v_typ/U0:.2e}  (参考径向 {v_correct_typ/U0:.2e})

    对力的影响 (假设 ~{N_neighbors} 个邻居粒子贡献, 量级估算):
      DRAG_COEFF = {drag_coeff:.4e} N/(m/s)
      F_wake ~ DRAG * {N_neighbors} * v_wake ~ {drag_coeff * N_neighbors * v_typ:.4e} N

    对加速度的影响 (量级估算):
      a_wake = F/m ~ {drag_coeff * N_neighbors * v_typ / mass_p:.2e} m/s^2  (~{drag_coeff * N_neighbors * v_typ / mass_p / 9.81:.2f} g)""")

    print("\n" + "=" * 78)
    print("  结论: CHAR_LENGTH_M 已与粒子半径对齐；尾流速度按 |source_vel| 缩放。")
    print("  参考工况下 Re_D = Re_closure = 1；改变 |U| 或 ν 时闭式 Re 可另行传入。")
    print("=" * 78)


if __name__ == "__main__":
    print_dimensional_analysis()
