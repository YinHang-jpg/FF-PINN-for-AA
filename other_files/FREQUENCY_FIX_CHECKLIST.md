# 频率独立性修复验证清单

## 已修复的问题

### 1. UNIFIED_PINN.py
- ✅ `PhysicalUnifiedModel.__init__`: 从归一化参数 t_max 推断频率，保存为 `self.inferred_frequency`
- ✅ `PhysicalUnifiedModel.forward`: 使用 `self.inferred_frequency` 计算 omega (第277行)
- ✅ `load_individual_models`: 从 model_base_path 推断频率，用于创建 ARFNetT 和 StokesNetT 实例 (第308-340行, 第381-383行)

### 2. freq_sweep_with_training.py
- ✅ `run_script`: 改用 `subprocess.run([python, script.py])` 而不是 `exec()`，确保每次训练都启动新的Python进程，重新导入frequency变量 (第120-127行)

### 3. freq_sweep.py  
- ✅ `run_simulation_for_frequency`: 根据频率动态计算domain_size = 2 * wavelength (第209-218行)
- ✅ `run_simulation_for_frequency`: 根据domain_size动态设置绘图范围 (第260-264行)

## 为什么现在应该可以工作

### 训练阶段
1. `freq_sweep_with_training.py` 在处理每个频率前：
   - 恢复原始频率配置文件
   - 更新配置文件中的 frequency 为目标频率
   - 使用 subprocess.run 启动新的Python进程运行训练脚本
   
2. 训练脚本 (ARF_PINN_x.py等) 在新进程中：
   - 重新导入 frequency 变量（已被更新）
   - 使用正确的 frequency 值：
     - 计算 wavelength 和 k 用于初始化模型结构
     - 生成训练数据
     - 保存的模型权重（包括fourier_weights）对应正确的频率

### 推理阶段
1. `freq_sweep.py` 加载模型时：
   - 通过 `load_physical_unified_model` 加载指定频率文件夹的模型
   - `load_individual_models` 从路径推断频率 (如 'freq_8k' → 8000 Hz)
   - 使用推断的频率创建模型实例（用于设置period）
   - 加载保存的权重（包括训练时生成的fourier_weights）
   
2. `PhysicalUnifiedModel` 进行力计算时：
   - 从归一化参数的 t_max 推断频率
   - 使用推断的频率计算 omega 和 wavelength
   - 确保力计算使用正确的频率参数

3. 粒子仿真：
   - 域大小根据频率动态调整（2倍波长）
   - 绘图范围也相应调整
   - 不同频率的聚集位置会明显不同

## 剩余的非问题

### UNIFIED_PINN.py 中的全局 frequency 使用
- 第54, 74行：仅作为 get() 的默认值，不影响实际运行
- 第173-175行：ARFNet2D 类未被使用
- 第494-596行：测试/可视化函数，不影响实际训练和推理

### 训练脚本中的全局 frequency 导入
- ARF_PINN_x.py, ARF_PINN_t.py, STOKES_PINN_x.py, STOKES_PINN_t.py 等都导入了全局 frequency
- **这是正确的！** 因为：
  1. 训练前会更新配置文件中的 frequency
  2. 使用 subprocess.run 启动新进程，重新导入
  3. 每个频率的训练都在独立进程中，不会互相干扰

## 验证步骤

1. 删除所有旧模型：
   ```bash
   python results\parameter_sweep\clean_freq_models.py
   ```

2. 重新训练（确保使用正确的frequency）：
   ```bash
   python results\parameter_sweep\freq_sweep_with_training.py
   ```
   
   预期输出：
   - 每个频率训练前会显示"更新频率: X Hz"
   - 训练数据生成会使用对应的wavelength
   - 模型保存到 PINN/freq_Xk/ 文件夹

3. 运行仿真（使用训练好的模型）：
   ```bash
   python results\parameter_sweep\freq_sweep.py
   ```
   
   预期输出：
   - 每个频率显示"从路径推断频率: X Hz"
   - 每个频率显示"从归一化参数推断频率: X Hz"
   - 域大小根据频率不同（8kHz: 85mm, 10kHz: 68mm, 12kHz: 56.67mm, 16kHz: 42.5mm）
   - 聚集位置明显不同
   - 图像中四条曲线的峰值位置应该不同

## 如果还有问题

可能的原因：
1. 旧模型没有删除干净 → 重新运行 clean_freq_models.py
2. Python缓存没有清除 → 重新运行 clear_cache.py
3. 配置文件没有正确更新 → 手动检查 initialization/sound_source_standing.py 和 mechanisms/ARF.py 中的 frequency 值
