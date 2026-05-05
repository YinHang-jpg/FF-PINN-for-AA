"""DDPINN（Data-Driven PINN）训练配置。

正确流程（与用户描述一致）：
  Step 1: 与 wake_effect/PINN_wake 完全一致地分别训练 4 个子网络
          (Vr_PP, Vt_PP, Vr_Oseen, Vt_Oseen)，损失只来自解析公式；
  Step 2: 按 wake_effect/DEM_wake/DEM_wake.py 中的“分段混合 + 极坐标→笛卡尔”
          公式将 4 个子网络组合成 *统合模型* —— 此时该统合模型应与
          DEM_wake.acoustic_wake_velocity 数值一致；
  Step 3: 用 COMSOL 高保真度数据 (vx, vy)[m/s] 校准 *统合模型* 的 (vx, vy) 输出
          —— 单独看任一子网络的输出与 COMSOL 不应被强行对齐，所以校准只在
          “组合后”进行。

本文件给出 Step 3 的所有可调超参；Step 1 仍由 wake_effect/PINN_wake 完成，
DDPINN_finetune.py 会从 wake_effect/PINN_wake/PINN/（或 wake_effect/PINN/）
加载预训练权重作为微调初值。
"""
from __future__ import annotations

# ---------------- 用户对 COMSOL 数据的考察范围 ----------------
# 仅使用 r ≤ DOMAIN_RADIUS_M 的 COMSOL 节点，超出部分一律忽略。
DDPINN_DOMAIN_RADIUS_M = 5.0e-3  # 5 mm

# COMSOL 高保真算例对应的来流参考速度 |U| [m/s]
# 与 wake_effect/COMSOL_wake/compare_with_DEM/comsol_vs_DEM.py 默认 (-1, 0) 一致。
DDPINN_U_REF_M_S = 1.0

# ---------------- 微调超参 ----------------
# 数据损失：MSE((u_pred, v_pred)[m/s], (u_comsol, v_comsol)[m/s]) — 量纲已统一为 m/s。
# 物理正则：在 R∈[训练区间] 的随机采样上，每个子网络仍尽量贴近原解析公式，
# 防止子网络在 COMSOL 数据稀疏区域漂移过远。
DDPINN_W_DATA = 1.0
DDPINN_W_PHYS_REG = 1.0e-3

# 物理正则采样数（每次迭代都重新采样，覆盖完整训练区间）
DDPINN_N_PHYS_PER_NET = 4000

# 学习率与训练步数
DDPINN_LR = 1.0e-4
# 目标总损失。由于物理正则项有下限（COMSOL 与解析公式之间天然有偏差），
# 总损失的可达下界 ≈ W_PHYS_REG * L_phys_residual ≈ 1e-3 * O(1) ≈ 几 e-3。
# 设为 1e-2 让训练在数百 epoch 内自动收敛停下。
DDPINN_TARGET_LOSS = 1.0e-2
DDPINN_MAX_EPOCHS = 30000
DDPINN_PRINT_INTERVAL = 200
DDPINN_SCHEDULER_PATIENCE = 2000
