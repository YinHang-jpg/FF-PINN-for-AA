"""
Wake PINN 各子网络的无量纲 R 训练区间，与 DEM_wake 中 Oseen/PP 支路定义域一致。

R 为无量纲距离 r / CHAR_LENGTH_M（即 r / a，与 DEM_wake 相同）。
纯 PP 公式在分段中用于 R<=2；纯 Oseen 用于 R>=5。训练时在各支路有效区间内稠密采样。
"""
from __future__ import annotations

# 径向/切向 PP：拟合 Vr_PP、Vt_PP（DEM 中 r<=2a 用 PP 混合，公式本身在 R>0 有定义）
R_PP_R_MIN = 0.5
R_PP_R_MAX = 5.0

# 径向/切向 Oseen：拟合 Vr_Oseen、Vt_Oseen（R>=5a 用 Oseen；训练上界覆盖远场）
R_OSEEN_R_MIN = 2.0
R_OSEEN_R_MAX = 30.0
