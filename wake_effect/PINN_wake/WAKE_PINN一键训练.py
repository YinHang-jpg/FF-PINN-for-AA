#!/usr/bin/env python3
"""依次运行四个 Wake PINN 训练脚本（Vr/Vt × Oseen/PP）。

子进程使用 Matplotlib 非交互后端 (Agg)，训练过程中不弹出任何图片窗口。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SCRIPTS = (
    "WAKE_PINN_Vr_Oseen.py",
    "WAKE_PINN_Vr_PP.py",
    "WAKE_PINN_Vt_Oseen.py",
    "WAKE_PINN_Vt_PP.py",
)


def _configure_stdio_utf8() -> None:
    """避免本脚本在 Windows 默认代码页下 print 中文失败（子进程已通过环境变量处理）。"""
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
    # 避免子进程在 Windows 控制台因 cp1252 无法打印中文而崩溃
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for name in _SCRIPTS:
        script = root / name
        if not script.is_file():
            print(f"未找到脚本: {script}", file=sys.stderr)
            sys.exit(1)
        sep = "=" * 60
        print(f"\n{sep}\n>>> {name}\n{sep}\n", flush=True)
        proc = subprocess.run([py, str(script)], cwd=str(root), env=env)
        if proc.returncode != 0:
            print(f"\n[{name}] 退出码 {proc.returncode}，已中止。", file=sys.stderr)
            sys.exit(proc.returncode)
    print("\n四个模型训练流程已全部结束。")


if __name__ == "__main__":
    main()
