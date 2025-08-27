# 粒子运动模拟模型 - 核心脚本版本

这是一个干净的粒子运动模拟模型代码库，只包含核心的Python脚本文件，移除了所有大型数据文件、缓存文件和测试文件。

## 文件结构

```
├── main.py                          # 主仿真程序
├── requirements.txt                  # Python依赖包列表
├── .gitignore                       # Git忽略文件配置
├── mechanisms/                      # 物理机制模块
│   ├── ARF.py                      # 声辐射力计算
│   ├── Brownian.py                 # 布朗运动
│   ├── Stokes_drag.py              # 斯托克斯阻力
│   ├── agglomeration.py            # 粒子团聚
│   ├── collision.py                # 碰撞处理
│   ├── gravity.py                  # 重力
│   └── wake.py                     # 尾流场
└── initialization/                  # 初始化模块
    ├── particle_initialization.py   # 粒子初始化
    ├── sound_source_standing.py    # 驻波声源
    ├── sound_source_traveling.py   # 行波声源
    └── test_particle.py            # 测试粒子
```

## 主要功能

- **粒子运动仿真**：模拟500个粒子在声场中的运动
- **声辐射力**：计算并应用声辐射力到粒子上
- **多种物理机制**：支持重力、阻力、碰撞、团聚等
- **实时可视化**：动态显示粒子运动和密度分布
- **性能监控**：实时监控CPU、内存使用和帧率

## 安装和运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行仿真：
```bash
python main.py
```

## 配置选项

在 `main.py` 中可以调整以下开关：
- `USE_ARF`: 是否启用声辐射力
- `USE_GRAVITY`: 是否启用重力
- `USE_STOKES_DRAG`: 是否启用斯托克斯阻力
- `USE_COLLISION`: 是否启用碰撞处理
- `USE_ACOUSTIC_WAKE`: 是否启用尾流场影响

## 注意事项

- 此版本移除了所有大型PDF文件、数据文件和缓存文件
- 保留了核心的仿真逻辑和物理计算
- 适合代码审查、版本控制和部署
