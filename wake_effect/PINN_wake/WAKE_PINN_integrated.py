"""
WAKE_PINN_integrated.py — 包含 acoustic wake effect 的粒子动力学前向模拟

在 PINN_main_integrated.py 的基础上，额外引入粒子间声学尾流效应：
  - ARF (声辐射力) 由统一 PINN 模型给出
  - Stokes 阻力由统一 PINN 模型给出
  - Acoustic wake effect: 每个粒子在声场中振荡产生 Oseen/PP 型尾流场,
    修正邻近粒子所受 Stokes 阻力
  - 重力沿 y 方向

尾流计算提供两条路径:
  1. 解析公式 (Oseen + Proudman-Pearson 分段混合) — 默认
  2. 训练好的 4 个 Wake PINN 子模型 (若 .pth 文件存在)
"""

import sys
import os

# === 路径设置 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
_WAKE_EFFECT_DIR = os.path.dirname(SCRIPT_DIR)
_DEM_WAKE_DIR = os.path.join(_WAKE_EFFECT_DIR, "DEM_wake")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _DEM_WAKE_DIR not in sys.path:
    sys.path.insert(0, _DEM_WAKE_DIR)

from DEM_wake import (  # noqa: E402
    CHAR_LENGTH_M,
    WAKE_CLOSURE_RE_DEFAULT,
    WAKE_STRENGTH_MULTIPLIER,
    _A_P,
    _D_P,
    _MU_WATER,
)

if sys.platform.startswith('win'):
    import codecs
    if hasattr(sys.stdout, 'detach'):
        try:
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except (AttributeError, OSError):
            pass

import numpy as np
import torch
import torch.nn as nn
import json
import time as time_module
import pandas as pd

from mechanisms.UNIFIED_PINN import (
    load_individual_models,
    UnifiedNormalizer,
    PhysicalUnifiedModel,
)

from wake_theory_torch import (
    blend_vr_vt,
    dimless_vr_oseen,
    dimless_vr_pp,
    dimless_vt_oseen,
    dimless_vt_pp,
    polar_to_cartesian,
)

# ======================================================================
# 物理常数与可调参数（CHAR_LENGTH、闭式 Re、强度、粒径/粘度等与 DEM_wake.py 一致）
# ======================================================================
frequency = 10000        # Hz
gravity = 9.81           # m/s²

WAKE_RE = WAKE_CLOSURE_RE_DEFAULT  # Oseen/PP 闭式及 PINN 中 R*Re/4 等项的参量
WAKE_CUTOFF_R = 30.0                # 尾流截断无量纲半径
WAKE_UPDATE_INTERVAL = 10           # 每隔多少步刷新一次尾流

# Stokes 阻力系数（与 DEM_wake.print_dimensional_analysis 中 drag 定义一致）
_CUNNINGHAM = 1.0817
DRAG_COEFF = 3.0 * np.pi * _MU_WATER * _D_P / _CUNNINGHAM

# 本模块粒子几何/质量（与 DEM_wake 粒径一致；不采用 initialization 返回的直径与质量）
SIM_DOMAIN_SIZE_M = (0.034, 0.034)       # (Lx, Ly) [m]
SIM_PARTICLE_DENSITY_KG_M3 = 2000.0    # 固体密度 [kg/m³]，用于 m = ρ * 4/3 π a³，a = _A_P


def initial_positions_linear(N: int, domain_size: tuple[float, float]) -> np.ndarray:
    """直线排布（与原 initialize_particles(..., init_mode='linear') 几何一致）。"""
    Lx, Ly = domain_size
    x = np.linspace(0.0, Lx, N, endpoint=True)
    y = np.full(N, Ly / 2.0)
    return np.column_stack((x, y))


def particle_masses_sphere(N: int, density_kg_m3: float, radius_m: float) -> np.ndarray:
    """均匀球体质量 m = ρ * 4/3 π r³。"""
    m = float(density_kg_m3 * (4.0 / 3.0) * np.pi * radius_m ** 3)
    return np.full(N, m, dtype=np.float64)


# ======================================================================
# Wake PINN 网络定义 (结构必须与训练脚本一致)
# ======================================================================
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


# ======================================================================
# 模型加载
# ======================================================================
def load_wake_pinn_models(device):
    """加载 4 个 Wake PINN 子模型；若 .pth 不存在则返回 None（回退到解析公式）。"""
    wake_dir = os.path.join(os.path.dirname(SCRIPT_DIR), 'PINN')
    specs = {
        'Vr_PP':    ('wake_Vr_PP.pth',    'wake_Vr_PP_norm.json',    WakeNetVrPP),
        'Vt_PP':    ('wake_Vt_PP.pth',    'wake_Vt_PP_norm.json',    WakeNetVtPP),
        'Vr_Oseen': ('wake_Vr_Oseen.pth', 'wake_Vr_Oseen_norm.json', WakeNetVrOseen),
        'Vt_Oseen': ('wake_Vt_Oseen.pth', 'wake_Vt_Oseen_norm.json', WakeNetVtOseen),
    }
    models = {}
    for key, (pth_name, json_name, cls) in specs.items():
        pth_path = os.path.join(wake_dir, pth_name)
        json_path = os.path.join(wake_dir, json_name)
        if not os.path.exists(pth_path):
            print(f"  [Wake] {pth_name} not found — falling back to analytical formulas")
            return None
        if not os.path.exists(json_path):
            print(f"  [Wake] {json_name} not found — falling back to analytical formulas")
            return None
        with open(json_path, 'r') as f:
            params = json.load(f)
        model = cls()
        model.load_state_dict(
            torch.load(pth_path, map_location=device, weights_only=True))
        model.to(device).eval()
        models[key] = (model, params)
    print("  [Wake] 4 Wake PINN models loaded successfully")
    return models


def load_physical_unified_model(device):
    """加载 ARF+Stokes 统一 PINN 模型（路径使用绝对路径）。"""
    model_base_path = os.path.join(PROJECT_ROOT, 'PINN')

    print(f"\n{'='*70}")
    print("Loading physical unified model (ARF + Stokes) ...")
    print(f"  Model path: {model_base_path}")
    print(f"{'='*70}\n")

    models = load_individual_models(device, model_base_path)
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm':    os.path.join(model_base_path, 'arf_model_x_normalization_params.json'),
        'arf_t_norm':    os.path.join(model_base_path, 'arf_model_t_normalization_params.json'),
        'stokes_x_norm': os.path.join(model_base_path, 'stokes_model_x_normalization_params.json'),
        'stokes_t_norm': os.path.join(model_base_path, 'stokes_model_t_normalization_params.json'),
        'stokes_v_norm': os.path.join(model_base_path, 'stokes_model_v_normalization_params.json'),
    }
    unified_norm.load_from_individual_models(model_paths, device='cpu')
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
    return phys_model


def compute_force_using_physical_model(positions_t, velocities_t, t_scalar,
                                       phys_model, use_cuda=False):
    """ARF+Stokes 合力 Fx（仅 x 方向；y 方向重力在主循环中直接施加）。"""
    with torch.inference_mode():
        x = positions_t[:, 0]
        vx = velocities_t[:, 0]
        t_vec = torch.full_like(x, float(t_scalar))
        if use_cuda:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fx = phys_model(x, vx, t_vec)
        else:
            fx = phys_model(x, vx, t_vec)
        return fx.float() if fx.dtype != torch.float32 else fx


# ======================================================================
# 尾流速度计算 — 解析公式路径 (向量化, 分 chunk)
# ======================================================================
@torch.no_grad()
def compute_wake_velocity_analytical(positions_t, velocities_t, Re=WAKE_RE,
                                     cutoff_R=WAKE_CUTOFF_R,
                                     strength=WAKE_STRENGTH_MULTIPLIER,
                                     chunk_size=512):
    """
    Oseen/PP 解析公式向量化计算全部粒子的尾流诱导速度。
    对 target 分 chunk，避免 N×N 显存爆炸。

    与 DEM_wake.acoustic_wake_velocity 一致：无量纲 Vr/Vt 后经极坐标合成，
    再乘以 strength * |source_vel|（每个源粒子一行）得到 [m/s]。

    Returns
    -------
    wake_vx, wake_vy : Tensor (N,)
        每个粒子位置上的总尾流速度（所有邻近粒子贡献之和）。
    """
    N = positions_t.shape[0]
    device = positions_t.device
    all_x = positions_t[:, 0]
    all_y = positions_t[:, 1]
    src_speed = torch.linalg.norm(velocities_t, dim=1).clamp(min=0.0)

    wake_vx = torch.zeros(N, device=device)
    wake_vy = torch.zeros(N, device=device)
    cutoff_m = cutoff_R * CHAR_LENGTH_M

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tgt_x = all_x[start:end].unsqueeze(1)          # (C, 1)
        tgt_y = all_y[start:end].unsqueeze(1)

        dx = tgt_x - all_x.unsqueeze(0)                # (C, N) target − source
        dy = tgt_y - all_y.unsqueeze(0)

        r = torch.sqrt(dx * dx + dy * dy)
        R = r / CHAR_LENGTH_M

        valid = (r > 1e-12) & (r < cutoff_m) & (R > 0.5)
        R_safe = torch.where(valid, R, torch.ones_like(R))

        Theta = torch.atan2(dy, dx)

        Vr_Os = dimless_vr_oseen(R_safe, Theta, Re)
        Vt_Os = dimless_vt_oseen(R_safe, Theta, Re)
        Vr_PP = dimless_vr_pp(R_safe, Theta, Re)
        Vt_PP = dimless_vt_pp(R_safe, Theta, Re)
        vr, vt = blend_vr_vt(R_safe, Vr_PP, Vt_PP, Vr_Os, Vt_Os)

        # 极坐标 → 笛卡尔（Θ = arctan2(y,x)，与 DEM_wake 一致）
        U, V = polar_to_cartesian(vr, vt, Theta)

        pair_scale = strength * src_speed.unsqueeze(0)
        U = torch.where(valid & torch.isfinite(U), U, torch.zeros_like(U)) * pair_scale
        V = torch.where(valid & torch.isfinite(V), V, torch.zeros_like(V)) * pair_scale

        wake_vx[start:end] = U.sum(dim=1)
        wake_vy[start:end] = V.sum(dim=1)

    return wake_vx, wake_vy


# ======================================================================
# 尾流速度计算 — PINN 路径 (向量化, 分 chunk)
# ======================================================================
@torch.no_grad()
def compute_wake_velocity_pinn(positions_t, velocities_t, wake_models,
                               cutoff_R=WAKE_CUTOFF_R,
                               strength=WAKE_STRENGTH_MULTIPLIER,
                               chunk_size=512):
    """使用 4 个 Wake PINN 子模型计算尾流速度；量纲缩放与解析路径、DEM_wake 一致。"""
    N = positions_t.shape[0]
    device = positions_t.device
    all_x = positions_t[:, 0]
    all_y = positions_t[:, 1]
    src_speed = torch.linalg.norm(velocities_t, dim=1).clamp(min=0.0)

    m_VrPP, p_VrPP = wake_models['Vr_PP']
    m_VtPP, p_VtPP = wake_models['Vt_PP']
    m_VrOs, p_VrOs = wake_models['Vr_Oseen']
    m_VtOs, p_VtOs = wake_models['Vt_Oseen']

    cutoff_m = cutoff_R * CHAR_LENGTH_M
    wake_vx = torch.zeros(N, device=device)
    wake_vy = torch.zeros(N, device=device)

    def _pred(model, params, R_t, Th_t):
        R_s = torch.clamp(
            (R_t - params['R_min']) / (params['R_max'] - params['R_min']),
            0.0, 1.0)
        return (model(R_s, Th_t) * params['force_sigma'] + params['force_mu']).squeeze(-1)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        tgt_x = all_x[start:end].unsqueeze(1)
        tgt_y = all_y[start:end].unsqueeze(1)

        dx = tgt_x - all_x.unsqueeze(0)
        dy = tgt_y - all_y.unsqueeze(0)
        r = torch.sqrt(dx * dx + dy * dy)
        R = r / CHAR_LENGTH_M

        valid = (r > 1e-12) & (r < cutoff_m) & (R > 0.5)
        valid_idx = valid.nonzero(as_tuple=False)
        if valid_idx.shape[0] == 0:
            continue

        R_valid = R[valid_idx[:, 0], valid_idx[:, 1]].unsqueeze(1)
        Th_valid = torch.atan2(
            dy[valid_idx[:, 0], valid_idx[:, 1]],
            dx[valid_idx[:, 0], valid_idx[:, 1]],
        ).unsqueeze(1)

        vrPP = _pred(m_VrPP, p_VrPP, R_valid, Th_valid)
        vtPP = _pred(m_VtPP, p_VtPP, R_valid, Th_valid)
        vrOs = _pred(m_VrOs, p_VrOs, R_valid, Th_valid)
        vtOs = _pred(m_VtOs, p_VtOs, R_valid, Th_valid)

        R_flat = R_valid.squeeze(1)
        vr, vt = blend_vr_vt(R_flat, vrPP, vtPP, vrOs, vtOs)

        th_flat = Th_valid.squeeze(1)
        U, V = polar_to_cartesian(vr, vt, th_flat)

        src_j = valid_idx[:, 1]
        pair_scale = strength * src_speed[src_j]

        U = torch.where(torch.isfinite(U), U, torch.zeros_like(U)) * pair_scale
        V = torch.where(torch.isfinite(V), V, torch.zeros_like(V)) * pair_scale

        # scatter-add 到对应 target
        tgt_local = valid_idx[:, 0]
        chunk_vx = torch.zeros(end - start, device=device)
        chunk_vy = torch.zeros(end - start, device=device)
        chunk_vx.scatter_add_(0, tgt_local, U)
        chunk_vy.scatter_add_(0, tgt_local, V)

        wake_vx[start:end] += chunk_vx
        wake_vy[start:end] += chunk_vy

    return wake_vx, wake_vy


# ======================================================================
# 分析工具函数
# ======================================================================
def calculate_concentration_distribution(positions, x_grid_mm):
    """Gaussian kernel density estimation for concentration."""
    if len(positions) == 0:
        return np.zeros_like(x_grid_mm)
    x_mm = positions[:, 0] * 1000.0
    sigma = 0.3
    concentrations = np.zeros_like(x_grid_mm)
    for i, x_pos in enumerate(x_grid_mm):
        d = np.abs(x_mm - x_pos)
        concentrations[i] = np.sum(np.exp(-d ** 2 / (2 * sigma ** 2)))
    return concentrations


def calculate_displacement_distribution(initial_pos, final_pos, x_grid_mm):
    """Average displacement at each x position using Gaussian weights."""
    if len(initial_pos) == 0:
        return np.zeros_like(x_grid_mm)
    displacements = np.sqrt(
        (final_pos[:, 0] - initial_pos[:, 0]) ** 2
        + (final_pos[:, 1] - initial_pos[:, 1]) ** 2
    ) * 1000.0
    x_init_mm = initial_pos[:, 0] * 1000.0
    avg = np.zeros_like(x_grid_mm)
    sigma = 0.5
    for i, x_pos in enumerate(x_grid_mm):
        w = np.exp(-(x_init_mm - x_pos) ** 2 / (2 * sigma ** 2))
        s = w.sum()
        avg[i] = np.sum(w * displacements) / s if s > 0 else 0.0
    return avg


def save_positions_to_csv(position_data, output_path, num_particles=None):
    """Save particle x-positions to CSV (COMSOL-compatible format)."""
    times = position_data['times']
    positions_list = position_data['positions']
    total = len(positions_list[0])
    if num_particles is None:
        num_particles = total
    else:
        num_particles = min(num_particles, total)

    print(f"\nSaving positions to CSV ...")
    print(f"  Particles saved: {num_particles}/{total}")
    print(f"  Time points: {len(times)}")

    rows = []
    for t_idx, t_val in enumerate(times):
        rows.append([t_val] + positions_list[t_idx][:num_particles].tolist())
    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.2f} MB)")


# ======================================================================
# 主模拟函数
# ======================================================================
def run_single_simulation(
    total_steps,
    suppress_output=True,
    save_positions=False,
    position_interval=0.01,
    particle_density=SIM_PARTICLE_DENSITY_KG_M3,
    domain_size_m: tuple[float, float] | None = None,
    wake_enabled=True,
    wake_update_interval=WAKE_UPDATE_INTERVAL,
    wake_strength=WAKE_STRENGTH_MULTIPLIER,
    num_particles=4400,
):
    """
    运行带尾流效应的粒子动力学模拟。

    与 PINN_main_integrated.py 的 run_single_simulation 功能对齐，
    额外加入 acoustic wake effect。

    Returns
    -------
    如果 save_positions=False:  (x_grid, relative_change)
    如果 save_positions=True:   (x_grid, relative_change, position_data)
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    if not suppress_output:
        print(f"\n{'='*70}")
        print("WAKE_PINN Integrated Simulation")
        print(f"  Device:     {device}")
        print(f"  Particles:  {num_particles}")
        print(f"  Density:    {particle_density} kg/m^3")
        print(f"  Wake:       {'ON' if wake_enabled else 'OFF'}")
        if wake_enabled:
            print(f"  Wake update interval: every {wake_update_interval} steps")
            print(f"  Wake strength:        {wake_strength}")
        print(f"{'='*70}\n")

    # ---- 初始化粒子（布局线性；质量由 DEM 粒径 _A_P 与 particle_density 决定）----
    domain_size = domain_size_m if domain_size_m is not None else SIM_DOMAIN_SIZE_M
    positions_np = initial_positions_linear(num_particles, domain_size)
    velocities_np = np.zeros_like(positions_np)
    mass_np = particle_masses_sphere(num_particles, particle_density, _A_P)
    initial_positions = positions_np.copy()

    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)

    # ---- 加载模型 ----
    phys_model = load_physical_unified_model(device)

    wake_models = None
    use_pinn_wake = False
    if wake_enabled:
        print("\nLoading wake models ...")
        wake_models = load_wake_pinn_models(device)
        use_pinn_wake = wake_models is not None
        if not use_pinn_wake:
            print("  => Using analytical Oseen/PP formulas for wake")
        else:
            print("  => Using PINN models for wake")

    # ---- 时间步进参数 ----
    dt = 1e-6
    steps = total_steps
    max_sim_time = steps * dt

    if not suppress_output:
        print(f"\nStarting simulation: {steps} steps")
        print(f"  dt = {dt:.2e} s")
        print(f"  Total time = {max_sim_time:.6f} s ({max_sim_time * 1e3:.3f} ms)")

    total_start = time_module.perf_counter()

    position_records = []
    time_records = []
    next_save_time = 0.0

    cached_wake_vx = torch.zeros(num_particles, device=device)
    cached_wake_vy = torch.zeros(num_particles, device=device)

    with torch.inference_mode():
        inv_mass = 1.0 / mass_t
        sim_time = 0.0

        for step in range(steps):
            # -- 位置快照 --
            if save_positions and abs(sim_time - next_save_time) < dt / 2:
                pos_np = positions_t.detach().cpu().numpy()
                position_records.append((pos_np[:, 0] * 1000.0).copy())
                time_records.append(sim_time)
                if not suppress_output:
                    print(f"  Saved positions at t={sim_time:.4f}s (step {step})")
                next_save_time += position_interval

            # -- ARF + Stokes Fx --
            fx_current = compute_force_using_physical_model(
                positions_t, velocities_t, sim_time, phys_model, use_cuda)

            # -- Wake 更新 (每 wake_update_interval 步) --
            if wake_enabled and step % wake_update_interval == 0:
                if use_pinn_wake:
                    cached_wake_vx, cached_wake_vy = compute_wake_velocity_pinn(
                        positions_t, velocities_t, wake_models,
                        cutoff_R=WAKE_CUTOFF_R, strength=wake_strength)
                else:
                    cached_wake_vx, cached_wake_vy = compute_wake_velocity_analytical(
                        positions_t, velocities_t, Re=WAKE_RE,
                        cutoff_R=WAKE_CUTOFF_R, strength=wake_strength)

            # -- Forward Euler 时间推进 --
            positions_t.add_(velocities_t, alpha=dt)

            # x 方向: ARF+Stokes + wake drag
            if wake_enabled:
                fx_total = fx_current + DRAG_COEFF * cached_wake_vx
            else:
                fx_total = fx_current
            velocities_t[:, 0].add_(fx_total.mul(inv_mass), alpha=dt)

            # y 方向: gravity + wake drag
            if gravity > 0:
                velocities_t[:, 1].add_(-gravity * dt)
            if wake_enabled:
                velocities_t[:, 1].add_(
                    (DRAG_COEFF * cached_wake_vy).mul(inv_mass), alpha=dt)

            sim_time += dt

            # -- 进度输出 --
            if (not suppress_output) and step > 0 and step % 10000 == 0:
                elapsed = time_module.perf_counter() - total_start
                pct = step / steps * 100
                eta = elapsed / step * (steps - step)
                print(f"  Step {step}/{steps} ({pct:.1f}%) | "
                      f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | "
                      f"t_sim = {sim_time * 1e3:.3f} ms")

        # -- 保存最终位置 --
        if save_positions:
            final_t = steps * dt
            if len(time_records) == 0 or abs(time_records[-1] - final_t) > dt:
                pos_np = positions_t.detach().cpu().numpy()
                position_records.append((pos_np[:, 0] * 1000.0).copy())
                time_records.append(final_t)

    total_elapsed = time_module.perf_counter() - total_start
    if not suppress_output:
        print(f"\nSimulation completed: {total_elapsed:.2f}s")

    # ---- 浓度分析 ----
    positions_final = positions_t.detach().cpu().numpy()
    x_grid = np.linspace(2.0, 32.0, 500)

    init_conc = calculate_concentration_distribution(initial_positions, x_grid)
    final_conc = calculate_concentration_distribution(positions_final, x_grid)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = (final_conc - init_conc) / init_conc * 100.0
        rel_change = np.nan_to_num(rel_change, nan=0.0, posinf=0.0, neginf=0.0)

    if save_positions:
        position_data = {
            'times': np.array(time_records),
            'positions': position_records,
        }
        return x_grid, rel_change, position_data
    return x_grid, rel_change


# ======================================================================
# main — 直接运行入口
# ======================================================================
def main():
    """运行一次完整的带尾流效应仿真并打印统计结果。"""
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    num_particles = 4400
    particle_density = SIM_PARTICLE_DENSITY_KG_M3
    domain_size = SIM_DOMAIN_SIZE_M

    print(f"Device: {device}")
    print(f"Particles: {num_particles},  Density: {particle_density} kg/m^3")

    positions_np = initial_positions_linear(num_particles, domain_size)
    velocities_np = np.zeros_like(positions_np)
    mass_np = particle_masses_sphere(num_particles, particle_density, _A_P)
    initial_positions = positions_np.copy()

    positions_t = torch.as_tensor(positions_np, dtype=torch.float32, device=device)
    velocities_t = torch.as_tensor(velocities_np, dtype=torch.float32, device=device)
    mass_t = torch.as_tensor(mass_np, dtype=torch.float32, device=device)

    # 加载主力模型
    phys_model = load_physical_unified_model(device)

    # 加载尾流模型
    print("\nLoading wake models ...")
    wake_models = load_wake_pinn_models(device)
    use_pinn_wake = wake_models is not None
    if not use_pinn_wake:
        print("  => Analytical Oseen/PP formulas")
    else:
        print("  => PINN wake models")

    # 模拟参数
    dt = 1e-6
    steps = 100000
    wake_interval = WAKE_UPDATE_INTERVAL
    max_sim_time = steps * dt

    print(f"\n{'='*70}")
    print(f"Simulation: {steps} steps,  dt = {dt:.2e} s")
    print(f"  Total time: {max_sim_time * 1e3:.3f} ms")
    print(f"  Wake update: every {wake_interval} steps")
    print(f"  Wake strength: {WAKE_STRENGTH_MULTIPLIER}")
    print(f"{'='*70}\n")

    t0 = time_module.perf_counter()

    cached_wvx = torch.zeros(num_particles, device=device)
    cached_wvy = torch.zeros(num_particles, device=device)

    with torch.inference_mode():
        inv_mass = 1.0 / mass_t
        sim_time = 0.0

        for step in range(steps):
            fx = compute_force_using_physical_model(
                positions_t, velocities_t, sim_time, phys_model, use_cuda)

            if step % wake_interval == 0:
                if use_pinn_wake:
                    cached_wvx, cached_wvy = compute_wake_velocity_pinn(
                        positions_t, velocities_t, wake_models)
                else:
                    cached_wvx, cached_wvy = compute_wake_velocity_analytical(
                        positions_t, velocities_t)

            positions_t.add_(velocities_t, alpha=dt)

            fx_total = fx + DRAG_COEFF * cached_wvx
            velocities_t[:, 0].add_(fx_total.mul(inv_mass), alpha=dt)

            velocities_t[:, 1].add_(-gravity * dt)
            velocities_t[:, 1].add_(
                (DRAG_COEFF * cached_wvy).mul(inv_mass), alpha=dt)

            sim_time += dt

            if step > 0 and step % 10000 == 0:
                elapsed = time_module.perf_counter() - t0
                print(f"  Step {step}/{steps} ({step/steps*100:.1f}%) | "
                      f"{elapsed:.1f}s elapsed")

    total_elapsed = time_module.perf_counter() - t0
    print(f"\nSimulation completed: {total_elapsed:.2f}s")

    # 分析
    positions_final = positions_t.detach().cpu().numpy()
    x_grid = np.linspace(2.0, 32.0, 500)

    init_conc = calculate_concentration_distribution(initial_positions, x_grid)
    final_conc = calculate_concentration_distribution(positions_final, x_grid)
    disp_dist = calculate_displacement_distribution(
        initial_positions, positions_final, x_grid)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_change = (final_conc - init_conc) / init_conc * 100.0
        rel_change = np.nan_to_num(rel_change, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n=== Distribution Statistics ===")
    print(f"Relative change range: [{np.min(rel_change):.2f}%, "
          f"{np.max(rel_change):.2f}%]")
    print(f"Mean displacement: {np.mean(disp_dist):.4f} mm")
    print(f"Max displacement:  {np.max(disp_dist):.4f} mm")
    print("\nSimulation with acoustic wake effect completed successfully!")
    print("Use run_single_simulation() for programmatic access.")


if __name__ == "__main__":
    main()
