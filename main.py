import matplotlib.pyplot as plt
import matplotlib.animation as animation
from initialization.particle_initialization import initialize_particles
import numpy as np
import psutil
import time
import threading
import os
import torch
import json
import sys
from scipy.interpolate import PchipInterpolator

# 配置开关 - 使用PINN模型进行力预测，同时开启斯托克斯阻力
RUN_HEADLESS_BENCHMARK = True  # 关闭渲染与动画，仅计算并在1000步后退出
USE_TRAVELING_WAVE = False
USE_TEST_PARTICLE = False
USE_PINN_FORCES = True  # 使用PINN模型预测所有粒子受力
USE_GRAVITY = False
USE_STOKES_DRAG = True  # 开启斯托克斯阻力
USE_AGGLOMERATION = False
USE_BROWNIAN = False
USE_ACOUSTIC_WAKE = False  # 关闭，由PINN模型处理
USE_COLLISION = False  # 关闭碰撞处理

# 调试打印配置
TARGET_PARTICLE_INDEX = 0  # 要监控的粒子索引


# 仿真时间累计变量
simulation_time = 0.0  # 累计仿真时间
density_plot_saved = False  # 标记是否已经保存过密度图
last_save_time = 0.0  # 记录上次保存图片的时间

# 性能计时变量
step_timing_printed = False  # 标记是否已打印过第5帧的计时信息



# 导入声场计算（仅用于可视化）
if USE_TRAVELING_WAVE:
    from initialization.sound_source_traveling import compute_sound_field, frequency
    IS_STANDING_WAVE = False
else:
    from initialization.sound_source_standing import compute_sound_field, frequency
    IS_STANDING_WAVE = True

# 导入PINN模型
from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as NormalizerX
from mechanisms.ARF_PINN_t import ARFNetT as ARFNetT, Normalizer as NormalizerT

# 导入Stokes PINN模型
from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV

def compute_stokes_drag_force_using_pinn(positions, velocities, time, stokes_x_model, stokes_x_normalizer, stokes_t_model, stokes_t_normalizer, stokes_v_model, stokes_v_normalizer):
    """
    使用训练好的Stokes PINN模型计算斯托克斯阻力
    组合公式：F_stokes = F_v(vx) - F_x(x) * F_t(t)
    """
    if (stokes_x_model is None or stokes_t_model is None or stokes_v_model is None or
        stokes_x_normalizer is None or stokes_t_normalizer is None or stokes_v_normalizer is None):
        print("Stokes PINN模型未完全初始化，无法计算斯托克斯阻力")
        return np.zeros_like(positions)
    
    try:
        with torch.no_grad():
            # 准备输入数据
            x = torch.tensor(positions[:, 0], dtype=torch.float32)
            vx = torch.tensor(velocities[:, 0], dtype=torch.float32)
            t = torch.full_like(x, time, dtype=torch.float32)
            
            # 归一化输入 - 使用正确的键名
            # x模型输入：位置x
            x_min = stokes_x_normalizer.stats['x'][0]
            x_max = stokes_x_normalizer.stats['x'][1]
            x_norm = (x - x_min) / (x_max - x_min)
            x_norm = torch.clamp(x_norm, 0.0, 1.0)
            
            # t模型输入：时间t
            t_min = stokes_t_normalizer.stats['t'][0]
            t_max = stokes_t_normalizer.stats['t'][1]
            t_norm = (t - t_min) / (t_max - t_min)
            t_norm = torch.clamp(t_norm, 0.0, 1.0)
            
            # v模型输入：速度vx
            vx_min = stokes_v_normalizer.stats['vx'][0]
            vx_max = stokes_v_normalizer.stats['vx'][1]
            vx_norm = (vx - vx_min) / (vx_max - vx_min)
            vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
            
            # 使用各模型预测
            fx_spatial_norm = stokes_x_model(x_norm.unsqueeze(1))
            fx_temporal_norm = stokes_t_model(t_norm.unsqueeze(1))
            fx_velocity_norm = stokes_v_model(vx_norm.unsqueeze(1))
            
            # 反归一化 - 使用正确的键名
            fx_spatial_mu = stokes_x_normalizer.stats['fx'][0]
            fx_spatial_sigma = stokes_x_normalizer.stats['fx'][1]
            fx_spatial = fx_spatial_norm * fx_spatial_sigma + fx_spatial_mu
            
            fx_temporal_mu = stokes_t_normalizer.stats['fx'][0]
            fx_temporal_sigma = stokes_t_normalizer.stats['fx'][1]
            fx_temporal = fx_temporal_norm * fx_temporal_sigma + fx_temporal_mu
            
            fx_velocity_mu = stokes_v_normalizer.stats['fx'][0]
            fx_velocity_sigma = stokes_v_normalizer.stats['fx'][1]
            fx_velocity = fx_velocity_norm * fx_velocity_sigma + fx_velocity_mu
            
            # 组合公式：F_stokes = F_v(vx) - F_x(x) * F_t(t)
            # 注意：这里需要确保所有张量形状一致
            fx_stokes = fx_velocity.squeeze() - fx_spatial.squeeze() * fx_temporal.squeeze()
            fy_stokes = torch.zeros_like(fx_stokes)  # y方向为零
            
            # 转换为numpy数组
            fx_stokes = fx_stokes.numpy()
            fy_stokes = fy_stokes.numpy()
            
            return np.column_stack([fx_stokes, fy_stokes])
            
    except Exception as e:
        print(f"Stokes PINN模型计算阻力时出错: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros_like(positions)

def compute_particle_forces_using_pinn(positions, velocities, mass, radii, dt, time, x_model, x_normalizer, t_model, t_normalizer, stokes_x_model, stokes_x_normalizer, stokes_t_model, stokes_t_normalizer, stokes_v_model, stokes_v_normalizer, print_forces=False):
    """
    使用训练好的PINN模型计算粒子受力
    - x_model: 位置相关的力模型 (ARFNet)
    - t_model: 时间相关的力模型 (ARFNetT)
    - stokes_*_model: Stokes阻力相关模型
    - print_forces: 是否打印力信息
    """
    if x_model is None or x_normalizer is None or t_model is None or t_normalizer is None:
        print("ARF PINN模型未完全初始化，无法计算力")
        return positions, velocities, None
    
    try:
        with torch.no_grad():
            # 准备输入数据
            x = torch.tensor(positions[:, 0], dtype=torch.float32)
            y = torch.tensor(positions[:, 1], dtype=torch.float32)
            t = torch.full_like(x, time, dtype=torch.float32)
            
            # 将x坐标中心化到通道中点，再按训练范围归一化
            x_centered = x - (domain_size[0] / 2.0)
            x_norm = (x_centered - X_MIN) / (X_MAX - X_MIN)
            x_norm = torch.clamp(x_norm, 0.0, 1.0)
            
            # 归一化时间用于t模型
            t_min, t_max = 0.0, 0.0001  # 从归一化参数文件获取
            t_norm = (t - t_min) / (t_max - t_min)
            
            # 使用x模型预测位置相关的力（归一化后的值）
            fx_spatial_norm = x_model(x_norm.unsqueeze(1))
            
            # 使用t模型预测时间因子（归一化后的值）
            time_factor_norm = t_model(t_norm.unsqueeze(1))
            
            # 反归一化x模型的输出
            if x_normalizer is not None and 'fx' in x_normalizer.stats:
                fx_mu, fx_sigma = x_normalizer.stats['fx']
                fx_spatial = fx_spatial_norm * fx_sigma + fx_mu
            else:
                # 如果没有归一化器，使用加载/默认的反归一化参数
                fx_spatial = fx_spatial_norm * FORCE_SIGMA + FORCE_MU
            
            # 反归一化t模型的输出
            if t_normalizer is not None and 'fx' in t_normalizer.stats:
                tf_mu, tf_sigma = t_normalizer.stats['fx']
                time_factor = time_factor_norm * tf_sigma + tf_mu
            else:
                # 如果没有归一化器，使用默认的反归一化参数
                time_factor_mu = -0.0006397787947207689
                time_factor_sigma = 0.7075421214103699
                time_factor = time_factor_norm * time_factor_sigma + time_factor_mu
            
            # 组合得到ARF力：F_arf = F_spatial * F_temporal
            fx_arf = fx_spatial.squeeze() * time_factor.squeeze()
            fy_arf = torch.zeros_like(fx_arf)  # 假设只有x方向的力
            
            # 计算Stokes阻力（如果启用）
            if USE_STOKES_DRAG:
                stokes_force = compute_stokes_drag_force_using_pinn(
                    positions, velocities, time, stokes_x_model, stokes_x_normalizer, 
                    stokes_t_model, stokes_t_normalizer, stokes_v_model, stokes_v_normalizer
                )
                fx_stokes = stokes_force[:, 0]
                fy_stokes = stokes_force[:, 1]
            else:
                fx_stokes = np.zeros_like(fx_arf)
                fy_stokes = np.zeros_like(fx_arf)
            
            # 转换为numpy数组
            fx_arf = fx_arf.numpy()
            fy_arf = fy_arf.numpy()
            
            # 总力 = ARF力 + Stokes阻力
            fx_total = fx_arf + fx_stokes
            fy_total = fy_arf + fy_stokes
            
            # 打印力信息（如果需要）
            if print_forces:
                print(f"\n=== 时间步 {time:.6f}s 的粒子受力 ===")
                print("粒子ID | X位置(mm) | Y位置(mm) | ARF Fx(pN) | ARF Fy(pN) | Stokes Fx(pN) | Stokes Fy(pN) | 总Fx(pN) | 总Fy(pN)")
                print("-" * 140)
                for i in range(len(positions)):
                    print(f"{i:6d} | {positions[i,0]*1000:8.3f} | {positions[i,1]*1000:8.3f} | {fx_arf[i]*1e12:10.3e} | {fy_arf[i]*1e12:10.3e} | {fx_stokes[i]*1e12:12.3e} | {fy_stokes[i]*1e12:12.3e} | {fx_total[i]*1e12:9.3e} | {fy_total[i]*1e12:9.3e}")
                print(f"总粒子数: {len(positions)}")
                print(f"ARF 平均Fx: {np.mean(fx_arf)*1e12:.3e} pN | 范围: [{np.min(fx_arf)*1e12:.3e}, {np.max(fx_arf)*1e12:.3e}] pN")
                if USE_STOKES_DRAG:
                    print(f"Stokes 平均Fx: {np.mean(fx_stokes)*1e12:.3e} pN | 范围: [{np.min(fx_stokes)*1e12:.3e}, {np.max(fx_stokes)*1e12:.3e}] pN")
                print(f"总力 平均Fx: {np.mean(fx_total)*1e12:.3e} pN | 范围: [{np.min(fx_total)*1e12:.3e}, {np.max(fx_total)*1e12:.3e}] pN")
                print("=" * 140)
            
            # 计算加速度
            accelerations = np.column_stack([fx_total, fy_total]) / mass[:, None]
            
            # 更新速度和位置
            velocities = velocities + accelerations * dt
            positions = positions + velocities * dt
            
            return positions, velocities, (fx_total, fy_total)
            
    except Exception as e:
        print(f"PINN模型计算力时出错: {e}")
        print("将使用零力（无外力效应）")
        # 如果PINN计算失败，不施加任何力，保持原有运动
        return positions, velocities, None
# 移除所有力学计算模块的导入，现在只使用PINN模型

# 性能监控变量
performance_data = {
    'cpu_percent': 0.0,
    'memory_percent': 0.0,
    'memory_used_gb': 0.0,
    'fps': 0.0,
    'frame_time': 0.0,
    'particle_count': 0
}

# 性能监控函数
def update_performance_data():
    """更新性能数据"""
    while True:
        try:
            # CPU使用率
            performance_data['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率和具体使用量
            memory = psutil.virtual_memory()
            performance_data['memory_percent'] = memory.percent
            performance_data['memory_used_gb'] = memory.used / (1024**3)  # 转换为GB
            
            # 粒子数量
            performance_data['particle_count'] = len(positions) if 'positions' in globals() else 0
            
            time.sleep(0.5)  # 每0.5秒更新一次
        except:
            pass

# 启动性能监控线程
performance_thread = threading.Thread(target=update_performance_data, daemon=True)
performance_thread.start()



# 初始化粒子
domain_size = (0.034, 0.034)
positions, velocities, radii, mass = initialize_particles(N=10000, domain_size=domain_size)
# 记录初始位置用于位移计算
initial_positions = positions.copy()

# 初始化PINN模型
print("正在加载训练好的PINN模型...")
try:
    # 初始化ARF x模型（位置相关）
    x_model = ARFNetX(fourier_features=32)
    x_model_path = 'PINN/arf_model_x.pth'
    if os.path.exists(x_model_path):
        x_model.load_state_dict(torch.load(x_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载ARF x模型权重: {x_model_path}")
    else:
        print(f"警告: ARF x模型文件 {x_model_path} 不存在")
        x_model = None
    
    # 初始化ARF t模型（时间相关）
    t_model = ARFNetT(period_seconds=1.0 / frequency)
    t_model_path = 'PINN/arf_model_t.pth'
    if os.path.exists(t_model_path):
        t_model.load_state_dict(torch.load(t_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载ARF t模型权重: {t_model_path}")
    else:
        print(f"警告: ARF t模型文件 {t_model_path} 不存在")
        t_model = None
    
    # 初始化Stokes x模型（位置相关）
    stokes_x_model = StokesNetX(fourier_features=32)
    stokes_x_model_path = 'PINN/stokes_model_x.pth'
    if os.path.exists(stokes_x_model_path):
        stokes_x_model.load_state_dict(torch.load(stokes_x_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载Stokes x模型权重: {stokes_x_model_path}")
    else:
        print(f"警告: Stokes x模型文件 {stokes_x_model_path} 不存在")
        stokes_x_model = None
    
    # 初始化Stokes t模型（时间相关）
    stokes_t_model = StokesNetT(period_seconds=1.0 / frequency)
    stokes_t_model_path = 'PINN/stokes_model_t.pth'
    if os.path.exists(stokes_t_model_path):
        stokes_t_model.load_state_dict(torch.load(stokes_t_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载Stokes t模型权重: {stokes_t_model_path}")
    else:
        print(f"警告: Stokes t模型文件 {stokes_t_model_path} 不存在")
        stokes_t_model = None
    
    # 初始化Stokes v模型（速度相关）
    stokes_v_model = StokesNetV()
    stokes_v_model_path = 'PINN/stokes_model_v.pth'
    if os.path.exists(stokes_v_model_path):
        stokes_v_model.load_state_dict(torch.load(stokes_v_model_path, map_location='cpu', weights_only=True))
        print(f"成功加载Stokes v模型权重: {stokes_v_model_path}")
    else:
        print(f"警告: Stokes v模型文件 {stokes_v_model_path} 不存在")
        stokes_v_model = None
    
    # 加载归一化参数
    x_norm_path = 'PINN/arf_model_x_normalization_params.json'
    t_norm_path = 'PINN/arf_model_t_normalization_params.json'
    stokes_x_norm_path = 'PINN/stokes_model_x_normalization_params.json'
    stokes_t_norm_path = 'PINN/stokes_model_t_normalization_params.json'
    stokes_v_norm_path = 'PINN/stokes_model_v_normalization_params.json'
    
    x_normalizer = None
    t_normalizer = None
    stokes_x_normalizer = None
    stokes_t_normalizer = None
    stokes_v_normalizer = None
    
    # 默认值（当文件缺失或读取失败时使用）
    X_MIN = -0.0255
    X_MAX = 0.0255
    FORCE_MU = 4.5863430241099845e-12
    FORCE_SIGMA = 1.4498783250382896e-11
    
    # 加载ARF模型归一化参数
    if os.path.exists(x_norm_path):
        x_normalizer = NormalizerX()
        x_normalizer.load(x_norm_path, device='cpu')
        try:
            with open(x_norm_path, 'r') as f:
                _params = json.load(f)
            if 'x_min' in _params and 'x_max' in _params:
                X_MIN = float(_params['x_min'])
                X_MAX = float(_params['x_max'])
            if 'force_mu' in _params:
                FORCE_MU = float(_params['force_mu'])
            if 'force_sigma' in _params:
                FORCE_SIGMA = max(float(_params['force_sigma']), 1e-30)
            print(f"成功加载ARF x模型归一化参数: {x_norm_path}")
            print(f"x范围: [{X_MIN*1000:.1f}, {X_MAX*1000:.1f}] mm | 力均值/方差: {FORCE_MU:.3e}/{FORCE_SIGMA:.3e}")
        except Exception as _e:
            print(f"读取ARF x归一化参数文件时出错: {_e}，使用默认参数")
    
    if os.path.exists(t_norm_path):
        t_normalizer = NormalizerT()
        t_normalizer.load(t_norm_path, device='cpu')
        print(f"成功加载ARF t模型归一化参数: {t_norm_path}")
    
    # 加载Stokes模型归一化参数
    if os.path.exists(stokes_x_norm_path):
        stokes_x_normalizer = StokesNormalizerX()
        # 手动加载并转换格式
        with open(stokes_x_norm_path, 'r') as f:
            params = json.load(f)
        stokes_x_normalizer.stats = {
            'x': (torch.tensor(params['x_min'], dtype=torch.float32), 
                  torch.tensor(params['x_max'] - params['x_min'], dtype=torch.float32)),
            'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                   torch.tensor(params['force_sigma'], dtype=torch.float32))
        }
        print(f"成功加载Stokes x模型归一化参数: {stokes_x_norm_path}")
    
    if os.path.exists(stokes_t_norm_path):
        stokes_t_normalizer = StokesNormalizerT()
        # 手动加载并转换格式
        with open(stokes_t_norm_path, 'r') as f:
            params = json.load(f)
        stokes_t_normalizer.stats = {
            't': (torch.tensor(params['t_min'], dtype=torch.float32), 
                  torch.tensor(params['t_max'] - params['t_min'], dtype=torch.float32)),
            'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                   torch.tensor(params['force_sigma'], dtype=torch.float32))
        }
        print(f"成功加载Stokes t模型归一化参数: {stokes_t_norm_path}")
    
    if os.path.exists(stokes_v_norm_path):
        stokes_v_normalizer = StokesNormalizerV()
        # 手动加载并转换格式
        with open(stokes_v_norm_path, 'r') as f:
            params = json.load(f)
        stokes_v_normalizer.stats = {
            'vx': (torch.tensor(params['vx_min'], dtype=torch.float32), 
                   torch.tensor(params['vx_max'] - params['vx_min'], dtype=torch.float32)),
            'fx': (torch.tensor(params['force_mu'], dtype=torch.float32), 
                   torch.tensor(params['force_sigma'], dtype=torch.float32))
        }
        print(f"成功加载Stokes v模型归一化参数: {stokes_v_norm_path}")
    
    # 设置为评估模式
    if x_model is not None:
        x_model.eval()
    if t_model is not None:
        t_model.eval()
    if stokes_x_model is not None:
        stokes_x_model.eval()
    if stokes_t_model is not None:
        stokes_t_model.eval()
    if stokes_v_model is not None:
        stokes_v_model.eval()
    
    print("所有PINN模型初始化完成")
    
except Exception as e:
    print(f"PINN模型初始化失败: {e}")
    print("将无法进行粒子力预测")
    x_model = None
    t_model = None
    x_normalizer = None
    t_normalizer = None
    stokes_x_model = None
    stokes_t_model = None
    stokes_v_model = None
    stokes_x_normalizer = None
    stokes_t_normalizer = None
    stokes_v_normalizer = None

# 移除测试粒子相关代码，专注于PINN模型预测

if RUN_HEADLESS_BENCHMARK:
    # 仅运行计算，不进行任何渲染或动画，累计指定步数后退出并打印耗时
    steps = 10000
    dt = 1e-6
    print(f"Headless benchmark started: steps={steps}, dt={dt}")
    print("开始计算...")
    
    # 整体计算时间计时
    total_start_time = time.perf_counter()
    bench_t0 = time.perf_counter()
    
    for step in range(steps):
        # 推进时间
        simulation_time += dt
        t = simulation_time

        # 力学更新：PINN（包含ARF和Stokes阻力）
        if USE_PINN_FORCES and (x_model is not None and t_model is not None):
            positions, velocities, _ = compute_particle_forces_using_pinn(
                positions, velocities, mass, radii, dt, t, x_model, x_normalizer, t_model, t_normalizer, 
                stokes_x_model, stokes_x_normalizer, stokes_t_model, stokes_t_normalizer, 
                stokes_v_model, stokes_v_normalizer, print_forces=False
            )
        
        # 每10000步打印一次进度
        if (step + 1) % 10000 == 0:
            current_time = time.perf_counter()
            elapsed_so_far = current_time - total_start_time
            progress = (step + 1) / steps * 100
            print(f"进度: {step + 1}/{steps} ({progress:.1f}%) - 已用时: {elapsed_so_far:.2f}s")

    # 计算总耗时
    total_end_time = time.perf_counter()
    total_elapsed = total_end_time - total_start_time
    bench_t1 = time.perf_counter()
    physics_elapsed = bench_t1 - bench_t0
    
    print(f"\n=== 计算完成 ===")
    print(f"总步数: {steps}")
    print(f"时间步长: {dt:.2e} s")
    print(f"总计算时间: {total_elapsed:.4f} s ({total_elapsed/60:.2f} 分钟)")
    print(f"平均每步耗时: {total_elapsed/steps*1e3:.3f} ms")
    print(f"物理计算时间: {physics_elapsed:.4f} s")
    print(f"最终仿真时间: {simulation_time:.6f} s")

    # 计算完成后，绘制一次静态图（粒子位置 + 密度分布），不保存，仅展示
    def _calculate_concentration_distribution_h(positions, x_grid):
        """基于粒子到参考位置的距离计算浓度分布（headless版本）"""
        try:
            current_positions = positions[:, 0] * 1000.0
            current_positions = current_positions[np.isfinite(current_positions)]
            
            if len(current_positions) == 0:
                return np.zeros_like(x_grid)
            
            concentrations = np.zeros_like(x_grid)
            
            for i, x_pos in enumerate(x_grid):
                distances = np.abs(current_positions - x_pos)
                sigma = 0.2  # 浓度分布的标准差（mm）
                concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
                concentrations[i] = concentration
            
            return concentrations
        except Exception as e:
            print(f"浓度计算错误: {e}")
            return np.zeros_like(x_grid)

    fig_h, (ax_p_h, ax_d_h) = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：粒子散点（单位mm）
    ax_p_h.set_xlim(0, domain_size[0] * 1000)
    ax_p_h.set_ylim(0, domain_size[1] * 1000)
    ax_p_h.set_title("Particle Positions (after computation)")
    ax_p_h.set_xlabel("x (mm)")
    ax_p_h.set_ylabel("y (mm)")
    ax_p_h.scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, s=(radii * 1e6 * 4)**2, c='blue', alpha=0.6)

    # 右图：密度分布（使用新的浓度计算方式）
    x_grid_h = np.linspace(0.0, domain_size[0] * 1000.0, 100)
    initial_concentrations_h = _calculate_concentration_distribution_h(initial_positions, x_grid_h)
    current_concentrations_h = _calculate_concentration_distribution_h(positions, x_grid_h)
    
    # 计算相对变化
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change_h = (current_concentrations_h - initial_concentrations_h) / initial_concentrations_h * 100.0
        relative_change_h = np.nan_to_num(relative_change_h, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 绘制柱状图和平滑曲线
    ax_d_h.bar(x_grid_h, relative_change_h, width=0.34, align='center', alpha=0.8, label='Concentration Change')
    
    # 添加平滑曲线
    try:
        pchip = PchipInterpolator(x_grid_h, relative_change_h, extrapolate=False)
        x_smooth_h = np.linspace(x_grid_h[0], x_grid_h[-1], max(50, 4 * len(x_grid_h)))
        y_smooth_h = pchip(x_smooth_h)
        ax_d_h.plot(x_smooth_h, y_smooth_h, 'r-', linewidth=2, label='Smoothed')
    except Exception:
        ax_d_h.plot(x_grid_h, relative_change_h, 'r-', linewidth=2, label='Smoothed')
    
    # 动态y轴范围
    if np.any(np.isfinite(relative_change_h)):
        y_min_h = float(np.nanmin(relative_change_h))
        y_max_h = float(np.nanmax(relative_change_h))
        y_margin_h = (y_max_h - y_min_h) * 0.1 if y_max_h > y_min_h else 1.0
        ax_d_h.set_ylim(y_min_h - y_margin_h, y_max_h + y_margin_h)
    
    ax_d_h.set_title("Particle Density Change Rate (after computation)")
    ax_d_h.set_xlabel("x (mm)")
    ax_d_h.set_ylabel("Relative Change (%)")
    ax_d_h.grid(True, alpha=0.3)
    ax_d_h.legend()

    plt.tight_layout()
    plt.show()

    sys.exit(0)

# 绘图初始化 - 创建两个子图
fig, (ax_particles, ax_density) = plt.subplots(1, 2, figsize=(16, 6))

# 左侧子图：粒子运动
ax_particles.set_xlim(0, domain_size[0] * 1000)
ax_particles.set_ylim(0, domain_size[1] * 1000)
ax_particles.set_title("Particle Motion")
ax_particles.set_xlabel("x (mm)")
ax_particles.set_ylabel("y (mm)")

# 右侧子图：密度变化曲线
ax_density.set_xlim(0, domain_size[0] * 1000)
ax_density.set_ylim(-5, 5)  # 进一步缩小y轴范围，确保曲线可见
ax_density.set_title("Particle Density Distribution")
ax_density.set_xlabel("x (mm)")
ax_density.set_ylabel("Relative Change (%)")
ax_density.grid(True, alpha=0.3)

# 移除测试粒子图例

# 声场
X, Y, P = compute_sound_field(domain_size=domain_size, resolution=(200, 200), time=0.0)
# 根据初始声压场动态设置可视化幅值范围
_p_amp = float(np.max(np.abs(P))) if np.size(P) > 0 else 1.0
sound_img = ax_particles.imshow(
    P,
    extent=(0, domain_size[0] * 1000, 0, domain_size[1] * 1000),
    origin='lower',
    cmap='RdBu_r',
    alpha=0.4,
    vmin=-_p_amp,
    vmax=_p_amp
)

# 普通粒子
scatter = ax_particles.scatter(positions[:, 0] * 1000,
                     positions[:, 1] * 1000,
                     s=(radii * 1e6 * 4)**2,
                     c='blue', alpha=0.6)

# 移除测试粒子和尾流箭头相关代码

# 添加性能信息文本框
performance_text = ax_particles.text(0.02, 0.98, '', transform=ax_particles.transAxes, 
                          verticalalignment='top', fontsize=10,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 初始化基于距离的浓度统计机制（参考comsol_visualizer.py）
# 三个参考位置：x=0mm, x=8.5mm, x=25.5mm
reference_positions = np.array([0.0, 8.5, 25.5])  # mm
num_references = len(reference_positions)

# 浓度统计参数
concentration_scale = 100.0  # 浓度缩放因子
distance_weight = 1.0  # 距离权重
baseline_concentration = 0.0  # 基准浓度（相对变化）

# 创建x轴网格用于显示浓度分布
x_grid = np.linspace(0.0, domain_size[0] * 1000.0, 100)  # 100个点用于平滑显示
concentration_values = np.zeros_like(x_grid)

def calculate_concentration_distribution(positions, x_grid):
    """基于粒子到参考位置的距离计算浓度分布"""
    try:
        # 获取当前帧的粒子位置（单位：mm）
        current_positions = positions[:, 0] * 1000.0
        current_positions = current_positions[np.isfinite(current_positions)]
        
        if len(current_positions) == 0:
            return np.zeros_like(x_grid)
        
        # 计算每个x_grid位置处的浓度
        concentrations = np.zeros_like(x_grid)
        
        for i, x_pos in enumerate(x_grid):
            # 计算所有粒子到当前x位置的距离
            distances = np.abs(current_positions - x_pos)
            
            # 浓度计算：距离越近，浓度越高
            # 使用高斯函数形式的浓度分布
            sigma = 0.2  # 浓度分布的标准差（mm）
            concentration = np.sum(np.exp(-distances**2 / (2 * sigma**2)))
            
            concentrations[i] = concentration
        
        return concentrations
    except Exception as e:
        print(f"浓度计算错误: {e}")
        return np.zeros_like(x_grid)

# 计算初始浓度分布
initial_concentrations = calculate_concentration_distribution(positions, x_grid)
baseline_concentration = np.mean(initial_concentrations)

# 初始化柱状图和平滑曲线
bars = ax_density.bar(x_grid, [0]*len(x_grid), width=0.34, align='center', alpha=0.8, label='Concentration Change')
density_line, = ax_density.plot([], [], 'r-', linewidth=2, label='Smoothed')
ax_density.legend()



# 帧率计算变量
frame_times = []
last_frame_time = time.time()

def update(frame):
    global positions, velocities, radii, mass, frame_times, last_frame_time, density_plot_saved, last_save_time, initial_positions, step_timing_printed

    # 渲染间隔（仅控制可视化与UI的更新频率；物理仍每帧推进）
    render_interval = 1000
    should_render = (frame % render_interval == 0)

    # 开始计时（第5帧时）
    if frame == 5 and not step_timing_printed:
        t0 = time.perf_counter()

    # 计算帧率
    current_time = time.time()
    frame_time = current_time - last_frame_time
    frame_times.append(frame_time)
    
    # 保持最近30帧的时间记录
    if len(frame_times) > 30:
        frame_times.pop(0)
    
    # 计算平均帧率
    if len(frame_times) > 0:
        avg_frame_time = np.mean(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    else:
        fps = 0
    
    performance_data['fps'] = fps
    performance_data['frame_time'] = frame_time * 1000  # 转换为毫秒
    
    last_frame_time = current_time

    dt = 1e-6
    global simulation_time
    simulation_time += dt  # 累计仿真时间
    t = simulation_time  # 使用累计时间

    # 计时：声场计算开始
    if frame == 5 and not step_timing_printed:
        t1 = time.perf_counter()

    # 使用PINN模型计算粒子受力（包含ARF和Stokes阻力）
    if USE_PINN_FORCES:
        if x_model is not None and t_model is not None:
            positions, velocities, forces = compute_particle_forces_using_pinn(
                positions, velocities, mass, radii, dt, t, x_model, x_normalizer, t_model, t_normalizer, 
                stokes_x_model, stokes_x_normalizer, stokes_t_model, stokes_t_normalizer, 
                stokes_v_model, stokes_v_normalizer, print_forces=False
            )
        else:
            # 暂时禁用警告打印
            # if frame % 100 == 0:  # 每100帧打印一次状态
            #     print("警告: PINN模型未完全初始化，跳过力计算")
            pass  # 不施加任何力，保持原有运动
    
    # 计时：PINN计算结束，可视化更新开始
    if frame == 5 and not step_timing_printed:
        t2 = time.perf_counter()

    # 更新声场 & 粒子位置（仅在需要渲染的帧进行可视化更新）
    if should_render:
        Nx, Ny = (200, 200)
        _, _, P = compute_sound_field(domain_size=domain_size, resolution=(Nx, Ny), time=t)
        sound_img.set_data(P)
        scatter.set_offsets(positions * 1000)
    
    # 计时：声场和散点图更新结束，密度计算开始
    if frame == 5 and not step_timing_printed:
        t3 = time.perf_counter()
    
    if should_render:
        # 更新基于距离的浓度分布
        current_concentrations = calculate_concentration_distribution(positions, x_grid)
        
        # 计算相对变化
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_change = (current_concentrations - initial_concentrations) / initial_concentrations * 100.0
            relative_change = np.nan_to_num(relative_change, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新每个bar的高度
        for rect, h in zip(bars, relative_change):
            rect.set_height(h)

        # 更新平滑曲线（PCHIP 连接柱顶）
        try:
            pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
            x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
            y_smooth = pchip(x_smooth)
            density_line.set_data(x_smooth, y_smooth)
        except Exception:
            density_line.set_data(x_grid, relative_change)

        # 动态调整y轴范围
        if np.any(np.isfinite(relative_change)):
            y_min = float(np.nanmin(relative_change))
            y_max = float(np.nanmax(relative_change))
            y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
            ax_density.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax_density.set_title(f"Particle Density Change Rate (t = {t:.6f} s)")
    
    # 计时：密度计算结束，性能显示开始
    if frame == 5 and not step_timing_printed:
        t4 = time.perf_counter()


    
    # 每0.01s保存一次密度分布图，从0.01s到0.2s（仅在渲染帧进行保存与绘制）
    if should_render:
        if t >= 0.01 and t <= 0.2 and (t - last_save_time) >= 0.01:
            # 创建密度分布图（使用新的浓度计算方式）
            plt.figure(figsize=(12, 8))
            plt.bar(x_grid, relative_change, width=0.34, align='center', alpha=0.8, label='Concentration Change')
            
            # 添加平滑曲线
            try:
                pchip = PchipInterpolator(x_grid, relative_change, extrapolate=False)
                x_smooth = np.linspace(x_grid[0], x_grid[-1], max(50, 4 * len(x_grid)))
                y_smooth = pchip(x_smooth)
                plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed')
            except Exception:
                plt.plot(x_grid, relative_change, 'r-', linewidth=2, label='Smoothed')
            
            plt.xlabel('x (mm)')
            plt.ylabel('Relative Change (%)')
            plt.title(f'Particle Density Change Rate at t = {t:.6f} s')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 保存图片，文件名包含时间信息
            filename = f'density_plot_{int(t*1000):03d}_t_{t:.6f}s.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            last_save_time = t  # 更新上次保存时间

    # 更新性能信息显示
    arf_status = "✓ ARF Active" if (x_model is not None and t_model is not None) else "✗ ARF Inactive"
    stokes_status = "✓ Stokes Active" if (stokes_x_model is not None and stokes_t_model is not None and stokes_v_model is not None) else "✗ Stokes Inactive"
    if should_render:
        performance_text.set_text(
            f'Simulation Time: {t:.6f} s\n'
            f'FPS: {performance_data["fps"]:.1f}\n'
            f'Frame Time: {performance_data["frame_time"]:.1f}ms\n'
            f'Particles: {performance_data["particle_count"]}\n'
            f'ARF Models: {arf_status}\n'
            f'Stokes Models: {stokes_status}'
        )
    
    # 计时：性能显示结束
    if frame == 5 and not step_timing_printed:
        t5 = time.perf_counter()
    
    # 触发一次立即绘制并测量（仅在第5帧做一次）
    if frame == 5 and not step_timing_printed:
        pending_draw_measurement = {'draw_start': time.perf_counter(), 'draw_end': None}
        fig.canvas.draw()
        if pending_draw_measurement['draw_end'] is None:
            # 后备：立刻取一次时间，包含渲染阻塞
            pending_draw_measurement['draw_end'] = time.perf_counter()

        def ms(x):
            return max(0.0, x) * 1000.0

        # 计算各阶段耗时（缺失时间戳用相邻前一刻兜底，避免未定义）
        t1_l = locals().get('t1', t0)
        t2_l = locals().get('t2', t1_l)
        t3_l = locals().get('t3', t2_l)
        t4_l = locals().get('t4', t3_l)
        t5_l = locals().get('t5', t4_l)

        t_sound = ms(t1_l - t0)
        t_arf = ms(t2_l - t1_l)
        t_scatter = ms(t3_l - t2_l)
        t_density = ms(t4_l - t3_l)
        t_perf = ms(t5_l - t4_l)
        draw_ms = ms(pending_draw_measurement['draw_end'] - pending_draw_measurement['draw_start'])
        total_ms = t_sound + t_arf + t_scatter + t_density + t_perf + draw_ms

        print("\nStep timing breakdown (frame 5):")
        print(f"  Particles:          {len(positions)}")
        print(f"  Sound field update: {t_sound:.2f} ms")
        print(f"  PINN computation:   {t_arf:.2f} ms")
        print(f"  Visualization:      {t_scatter:.2f} ms")
        print(f"  Density calculation:{t_density:.2f} ms")
        print(f"  Perf display:       {t_perf:.2f} ms")
        print(f"  Render (draw):      {draw_ms:.2f} ms")
        print(f"  Total (incl. draw): {total_ms:.2f} ms")
        step_timing_printed = True
    
    return sound_img, scatter, performance_text, density_line


ani = animation.FuncAnimation(fig, update, frames=20000, interval=0, blit=False)
plt.show()
