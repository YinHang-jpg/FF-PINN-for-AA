#!/usr/bin/env python3
"""DDPINN 一键训练入口。

DDPINN（Data-Driven PINN）的正确流程：
  Step 1  分别训练 4 个子 PINN（与 PINN_wake 完全一致，只用解析公式）
          —— 由 wake_effect/PINN_wake/WAKE_PINN一键训练.py 完成；
  Step 2  按 DEM_wake 公式（PP/Oseen 分段混合 + 极坐标→笛卡尔）把 4 个子网
          组合成一个统合模型；
  Step 3  以 COMSOL 高保真数据 (vx, vy)[m/s] 校准这个统合模型。

单独训练任一子网络去贴合 COMSOL 是错误的（COMSOL 是组合后的速度场，
任何单一子项都不该等于它）。因此本入口只串联 “Step 3” —— Step 1 请
事先在 PINN_wake 目录运行。本脚本会自动从 PINN_wake/PINN/ 加载预训练
权重作为微调初值；若缺失，DDPINN_finetune.py 会报错并提示。

子进程使用 Matplotlib 非交互后端 (Agg)，训练过程中不弹出图像窗口。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main() -> None:
    _configure_stdio_utf8()
    root = Path(__file__).resolve().parent
    py = sys.executable
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    script = root / "DDPINN_finetune.py"
    if not script.is_file():
        print(f"未找到脚本: {script}", file=sys.stderr)
        sys.exit(1)
    sep = "=" * 60
    print(f"\n{sep}\n>>> Step 3 — DDPINN_finetune.py（统合模型 + COMSOL 校准）\n{sep}\n", flush=True)
    proc = subprocess.run([py, str(script)], cwd=str(root), env=env)
    if proc.returncode != 0:
        print(f"\n[DDPINN_finetune.py] 退出码 {proc.returncode}，已中止。", file=sys.stderr)
        sys.exit(proc.returncode)
    print("\nDDPINN 微调流程已结束。")


if __name__ == "__main__":
    main()
