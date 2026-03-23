#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时运行脚本，用于启动密度扫描
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入并运行密度扫描脚本
if __name__ == "__main__":
    # 改变工作目录到项目根目录
    import os
    os.chdir(str(project_root))
    
    # 导入density_distribution_plot模块并运行main
    from results.density_sweep import density_distribution_plot
    density_distribution_plot.main()
