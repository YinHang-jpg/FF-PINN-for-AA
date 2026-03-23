# 粒子碰撞模块使用说明

## 概述

本模块实现了粒子系统的完全非弹性碰撞处理，包括碰撞检测、碰撞物理计算和粒子合并功能。

## 主要功能

### 1. 碰撞检测 (`detect_collisions`)
- 使用KDTree进行高效的近邻搜索
- 检测距离小于等于两粒子半径之和的粒子对
- 返回碰撞粒子对的索引列表

### 2. 对心碰撞判断 (`is_head_on_collision`)
- 判断两个粒子是否为对心碰撞
- 考虑粒子的相对位置和相对速度
- 预测即将发生的碰撞

### 3. 碰撞后速度计算 (`calculate_collision_velocity`)
- 基于动量守恒计算完全非弹性碰撞后的速度
- 碰撞后两粒子具有相同的速度

### 4. 粒子合并 (`merge_particles`)
- 合并碰撞的粒子对
- 计算合并后的位置、速度、半径和质量
- 保持体积守恒（半径按体积立方根计算）

### 5. 碰撞物理应用 (`apply_collision_physics`)
- 整合所有碰撞处理步骤
- 自动检测、计算和合并碰撞粒子

## 物理原理

### 完全非弹性碰撞
- 碰撞后两粒子结合为一个粒子
- 动量守恒：`m₁v₁ + m₂v₂ = (m₁ + m₂)v_final`
- 能量不守恒（部分动能转化为内能）

### 粒子合并规则
- **位置**：质量加权平均位置
- **速度**：动量守恒的最终速度
- **半径**：体积守恒的半径（`r³ = r₁³ + r₂³`）
- **质量**：两粒子质量之和

## 使用方法

### 在main.py中启用碰撞
```python
# 配置开关
USE_COLLISION = True  # 开启碰撞处理

# 在主循环中应用碰撞物理
if USE_COLLISION:
    positions, velocities, radii, mass = apply_collision_physics(
        positions, velocities, radii, mass, dt
    )
```

### 基本API调用
```python
from mechanisms.collision import apply_collision_physics

# 应用碰撞物理
new_positions, new_velocities, new_radii, new_mass = apply_collision_physics(
    positions, velocities, radii, mass, dt
)
```

## 参数说明

### 输入参数
- `positions`: 粒子位置数组 (N, 2)
- `velocities`: 粒子速度数组 (N, 2)
- `radii`: 粒子半径数组 (N,)
- `mass`: 粒子质量数组 (N,)
- `dt`: 时间步长

### 输出参数
- 更新后的位置、速度、半径和质量数组
- 碰撞后粒子数量可能减少（由于合并）

## 测试和验证

运行测试脚本验证功能：
```bash
python test_collision.py
```

测试包括：
- 碰撞检测功能
- 对心碰撞判断
- 碰撞后速度计算
- 粒子合并功能
- 完整碰撞物理流程
- 碰撞过程可视化

## 性能特点

- 使用KDTree进行高效的碰撞检测
- 支持大量粒子的实时碰撞处理
- 内存使用优化，避免重复计算
- 数值稳定性处理，避免除零错误

## 注意事项

1. **时间步长**：确保时间步长足够小，避免粒子穿透
2. **粒子密度**：高密度情况下可能产生多次碰撞，需要多次迭代
3. **边界条件**：当前版本假设粒子在无限空间中运动
4. **数值精度**：使用双精度浮点数确保计算精度

## 扩展功能

### 弹性碰撞
可以通过修改`calculate_collision_velocity`函数实现弹性碰撞：
```python
def calculate_elastic_collision_velocity(vel1, vel2, mass1, mass2):
    # 弹性碰撞公式
    # 动量守恒 + 动能守恒
    pass
```

### 部分非弹性碰撞
可以添加恢复系数参数：
```python
def calculate_partial_inelastic_collision(vel1, vel2, mass1, mass2, restitution=0.8):
    # 恢复系数控制碰撞后的能量损失
    pass
```

### 多粒子碰撞
当前版本处理成对碰撞，可以扩展为多粒子同时碰撞的处理。

## 故障排除

### 常见问题
1. **粒子穿透**：减小时间步长或增加碰撞检测频率
2. **性能问题**：粒子数量过多时，考虑空间分区或并行处理
3. **数值不稳定**：检查粒子初始条件，避免极端情况

### 调试信息
启用调试模式查看碰撞详情：
```python
# 获取碰撞统计信息
stats = calculate_merged_particle_properties(positions, velocities, radii, mass, collision_pairs)
print(f"碰撞次数: {stats['total_collisions']}")
print(f"能量损失: {stats['energy_loss']}")
```
