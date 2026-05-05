"""DDPINN 微调主脚本：用 COMSOL 高保真数据校准“统合的尾流模型”。

流程（与用户描述一致）：
  Step 1  Pre-train（已由 wake_effect/PINN_wake/WAKE_PINN一键训练.py 完成）
          产出 4 个子 PINN 权重：wake_Vr_PP.pth / wake_Vt_PP.pth /
          wake_Vr_Oseen.pth / wake_Vt_Oseen.pth；
  Step 2  Compose：CombinedWakeModel 把 4 个子网按 DEM_wake 公式（PP/Oseen 分段
          混合 + 极坐标→笛卡尔）拼接为单一可微分模型；此时该统合模型应与
          DEM_wake.acoustic_wake_velocity 数值近似一致；
  Step 3  Fine-tune：以 COMSOL 数据 (vx, vy)[m/s] 为目标 MSE，对统合模型
          (即 4 个子网络的 *联合* 参数) 反向传播。同时保留小幅的“与解析公式
          MSE”物理正则项，避免子网络在数据稀疏区域漂移。

注意（与用户三条提醒一一对应）：
  1) 量纲：COMSOL 单位 m/s，统合模型在前向中乘以 |U|=1 m/s 输出 m/s，二者
     直接做 MSE 即可，无需再做反归一化。
  2) 方向：所有“极坐标拆分”都在 CombinedWakeModel 内部完成，外部仅暴露
     笛卡尔接口，避免把 COMSOL (vx, vy) 与子网络极坐标输出错配。
  3) 范围：仅使用 r ≤ DDPINN_DOMAIN_RADIUS_M（默认 5 mm）的 COMSOL 节点。
"""
from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


SCRIPT_DIR = Path(__file__).resolve().parent
PINN_DIR = SCRIPT_DIR.parent
WAKE_EFFECT_DIR = PINN_DIR.parent
PROJECT_ROOT = WAKE_EFFECT_DIR.parent
DEM_WAKE_DIR = WAKE_EFFECT_DIR / "DEM_wake"
PINN_WAKE_DIR = WAKE_EFFECT_DIR / "PINN_wake"

for _p in (PROJECT_ROOT, WAKE_EFFECT_DIR, SCRIPT_DIR, DEM_WAKE_DIR, PINN_WAKE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from DEM_wake import (  # noqa: E402
    WAKE_CLOSURE_RE_DEFAULT as Re_FIXED,
    CHAR_LENGTH_M as A_P,
)
from wake_theory_torch import (  # noqa: E402
    dimless_vr_pp,
    dimless_vt_pp,
    dimless_vr_oseen,
    dimless_vt_oseen,
)
from wake_pinn_train_config import (  # noqa: E402
    R_PP_R_MIN,
    R_PP_R_MAX,
    R_OSEEN_R_MIN,
    R_OSEEN_R_MAX,
)

from ddpinn_combined_model import CombinedWakeModel, net_specs  # noqa: E402
from ddpinn_train_config import (  # noqa: E402
    DDPINN_DOMAIN_RADIUS_M,
    DDPINN_U_REF_M_S,
    DDPINN_W_DATA,
    DDPINN_W_PHYS_REG,
    DDPINN_N_PHYS_PER_NET,
    DDPINN_LR,
    DDPINN_TARGET_LOSS,
    DDPINN_MAX_EPOCHS,
    DDPINN_PRINT_INTERVAL,
    DDPINN_SCHEDULER_PATIENCE,
)
from ddpinn_comsol_data import load_comsol_polar  # noqa: E402


# 预训练权重的搜索路径（按优先级降序）
_PRETRAINED_DIRS = (
    PINN_WAKE_DIR / "PINN",
    WAKE_EFFECT_DIR / "PINN",
)


def find_pretrained_dir() -> Path:
    """寻找 4 个子网络 .pth + .json 都齐全的目录；优先 PINN_wake/PINN/。"""
    for d in _PRETRAINED_DIRS:
        if not d.is_dir():
            continue
        ok = all(
            (d / spec[2]).is_file() and (d / spec[3]).is_file() for spec in net_specs()
        )
        if ok:
            return d
    raise FileNotFoundError(
        "未找到完整的预训练子网络权重；请先运行 "
        "wake_effect/PINN_wake/WAKE_PINN一键训练.py 完成 Step 1。"
    )


def load_pretrained_into(combined: CombinedWakeModel, src_dir: Path, device) -> None:
    for name, _cls, json_name, pth_name in net_specs():
        with open(src_dir / json_name, "r", encoding="utf-8") as f:
            params = json.load(f)
        combined.set_norm(
            name,
            params["R_min"],
            params["R_max"],
            params.get("force_mu", 0.0),
            params.get("force_sigma", 1.0),
        )
        net = combined.get_subnet(name)
        net.load_state_dict(
            torch.load(src_dir / pth_name, map_location=device, weights_only=True)
        )


def main() -> None:
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"DDPINN 微调统合模型 — device={device}, Re={Re_FIXED}, "
        f"a={A_P:.3e} m, |U|_ref={DDPINN_U_REF_M_S} m/s"
    )

    # ---------------- 加载预训练（Step 1 已在 PINN_wake 完成） ----------------
    src_dir = find_pretrained_dir()
    print(f"加载预训练子网络: {src_dir}")
    combined = CombinedWakeModel(char_length_m=A_P).to(device)
    load_pretrained_into(combined, src_dir, device)

    # ---------------- 加载 COMSOL 数据（Step 3 的训练目标） ----------------
    print(
        f"加载 COMSOL 数据 (r ≤ {DDPINN_DOMAIN_RADIUS_M*1e3:.1f} mm, "
        f"|U|_ref={DDPINN_U_REF_M_S} m/s) …"
    )
    data = load_comsol_polar(
        char_length_m=A_P,
        u_ref_m_s=DDPINN_U_REF_M_S,
        r_max_m=DDPINN_DOMAIN_RADIUS_M,
    )
    n_data = int(data["x_m"].size)
    if n_data == 0:
        raise RuntimeError("COMSOL 在指定范围内无节点；请检查 r_max_m 设置。")
    print(f"  COMSOL 节点数: {n_data}")

    x_m_t = torch.tensor(data["x_m"].reshape(-1, 1), dtype=torch.float32, device=device)
    y_m_t = torch.tensor(data["y_m"].reshape(-1, 1), dtype=torch.float32, device=device)
    vx_t = torch.tensor(data["vx_m_s"].reshape(-1, 1), dtype=torch.float32, device=device)
    vy_t = torch.tensor(data["vy_m_s"].reshape(-1, 1), dtype=torch.float32, device=device)

    source_speed = float(DDPINN_U_REF_M_S)

    # ---------------- 微调前的基线评估 ----------------
    combined.eval()
    with torch.no_grad():
        u0, v0 = combined.forward_xy(x_m_t, y_m_t, source_speed)
        rmse0 = torch.sqrt(
            torch.mean((u0 - vx_t) ** 2 + (v0 - vy_t) ** 2)
        ).item()
        cos0 = float(torch.mean(
            (u0 * vx_t + v0 * vy_t)
            / (torch.sqrt(u0 ** 2 + v0 ** 2) * torch.sqrt(vx_t ** 2 + vy_t ** 2) + 1e-30)
        ).item())
    print(f"微调前 (u,v) RMSE vs COMSOL: {rmse0:.4f} m/s | mean cos = {cos0:.4f}")

    # ---------------- 优化器与损失 ----------------
    optimizer = optim.Adam(combined.parameters(), lr=DDPINN_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=DDPINN_SCHEDULER_PATIENCE
    )

    theory_fn = {
        "vr_pp": partial(dimless_vr_pp, Re=Re_FIXED),
        "vt_pp": partial(dimless_vt_pp, Re=Re_FIXED),
        "vr_oseen": partial(dimless_vr_oseen, Re=Re_FIXED),
        "vt_oseen": partial(dimless_vt_oseen, Re=Re_FIXED),
    }
    R_ranges = {
        "vr_pp": (R_PP_R_MIN, R_PP_R_MAX),
        "vt_pp": (R_PP_R_MIN, R_PP_R_MAX),
        "vr_oseen": (R_OSEEN_R_MIN, R_OSEEN_R_MAX),
        "vt_oseen": (R_OSEEN_R_MIN, R_OSEEN_R_MAX),
    }

    print(
        f"\n开始 fine-tune — W_DATA={DDPINN_W_DATA}, W_PHYS_REG={DDPINN_W_PHYS_REG}, "
        f"max_epochs={DDPINN_MAX_EPOCHS}, lr={DDPINN_LR}"
    )

    losses_total: list[float] = []
    losses_data: list[float] = []
    losses_phys: list[float] = []

    for epoch in range(DDPINN_MAX_EPOCHS + 1):
        combined.train()
        optimizer.zero_grad()

        # ---- 数据项: MSE 在 m/s 量纲下 ----
        u_pred, v_pred = combined.forward_xy(x_m_t, y_m_t, source_speed)
        loss_data = nn.functional.mse_loss(u_pred, vx_t) + nn.functional.mse_loss(
            v_pred, vy_t
        )

        # ---- 物理正则: 4 个子网络在归一化空间继续贴近原解析公式 ----
        loss_phys = torch.zeros((), device=device)
        for name in ("vr_pp", "vt_pp", "vr_oseen", "vt_oseen"):
            R_min, R_max = R_ranges[name]
            R_rand = (
                torch.rand(DDPINN_N_PHYS_PER_NET, 1, device=device) * (R_max - R_min)
                + R_min
            )
            T_rand = torch.rand(DDPINN_N_PHYS_PER_NET, 1, device=device) * (2.0 * np.pi)
            with torch.no_grad():
                target_dimless = theory_fn[name](R_rand, T_rand)
                mu = getattr(combined, f"{name}_mu")
                sigma = getattr(combined, f"{name}_sigma")
                target_norm = (target_dimless - mu) / sigma
            pred_norm = combined.predict_subnet_normalized(name, R_rand, T_rand)
            loss_phys = loss_phys + nn.functional.mse_loss(pred_norm, target_norm)

        loss = DDPINN_W_DATA * loss_data + DDPINN_W_PHYS_REG * loss_phys
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        losses_total.append(loss.item())
        losses_data.append(loss_data.item())
        losses_phys.append(loss_phys.item())

        if epoch % DDPINN_PRINT_INTERVAL == 0 or loss.item() < DDPINN_TARGET_LOSS:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:05d} | L = {loss.item():.3e} "
                f"| L_data = {loss_data.item():.3e} | L_phys = {loss_phys.item():.3e} "
                f"| LR = {lr:.2e}"
            )
        if loss.item() < DDPINN_TARGET_LOSS:
            print(f"达到目标损失 {DDPINN_TARGET_LOSS:.1e}，在第 {epoch} 个 epoch 停止训练")
            break

    # ---------------- 保存微调后的 4 子网络权重 ----------------
    save_dir = SCRIPT_DIR / "PINN"
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, _cls, json_name, pth_name in net_specs():
        torch.save(combined.get_subnet(name).state_dict(), save_dir / pth_name)
        n = combined.get_norm(name)
        with open(save_dir / json_name, "w", encoding="utf-8") as f:
            json.dump({**n, "Re": Re_FIXED}, f)
    print(f"\nDDPINN 4 个子网络权重 + 归一化参数已保存至: {save_dir}")

    # ---------------- 微调后的对比评估 ----------------
    combined.eval()
    with torch.no_grad():
        u_f, v_f = combined.forward_xy(x_m_t, y_m_t, source_speed)
        rmse_f = torch.sqrt(
            torch.mean((u_f - vx_t) ** 2 + (v_f - vy_t) ** 2)
        ).item()
        cos_f = float(torch.mean(
            (u_f * vx_t + v_f * vy_t)
            / (torch.sqrt(u_f ** 2 + v_f ** 2) * torch.sqrt(vx_t ** 2 + vy_t ** 2) + 1e-30)
        ).item())
    print(
        f"\n=== 与 COMSOL 对比 ===\n"
        f"  微调前 RMSE = {rmse0:.4f} m/s, mean_cos = {cos0:.4f}\n"
        f"  微调后 RMSE = {rmse_f:.4f} m/s, mean_cos = {cos_f:.4f}\n"
        f"  改进: ΔRMSE = {rmse0 - rmse_f:+.4f} m/s ({(1 - rmse_f/max(rmse0,1e-30))*100:+.1f}%)"
    )

    # ---------------- 训练曲线 ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses_total, label="total")
    ax.plot(losses_data, label="data", alpha=0.7)
    ax.plot(losses_phys, label="phys_reg", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DDPINN Fine-tune Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_png = SCRIPT_DIR / "DDPINN_finetune_loss.png"
    fig.savefig(out_png, dpi=160)
    print(f"训练曲线已保存: {out_png.name}")
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
