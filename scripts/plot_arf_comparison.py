import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from mechanisms.ARF_PINN import ARFNet, theoretical_arf

def load_model_and_params():
    """加载训练好的模型和归一化参数"""
    # 加载模型
    model = ARFNet(fourier_features=32)
    model.load_state_dict(torch.load('arf_pinn_model.pth', map_location='cpu'))
    model.eval()
    
    # 加载归一化参数
    with open('arf_pinn_normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    return model, norm_params

def predict_force(model, norm_params, x_values_mm):
    """使用训练好的模型预测力"""
    model.eval()
    with torch.no_grad():
        # 将毫米转换为米（模型训练时使用米）
        x_values_m = x_values_mm / 1000.0
        
        # 归一化输入
        x_min_m = norm_params['x_min']  # 米
        x_max_m = norm_params['x_max']  # 米
        x_scaled = (x_values_m - x_min_m) / (x_max_m - x_min_m)
        
        # 模型预测
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(1)
        pred_norm = model(x_tensor)
        
        # 反归一化输出
        force_mu = norm_params['force_mu']
        force_sigma = norm_params['force_sigma']
        pred_force = pred_norm * force_sigma + force_mu
        
        return pred_force.numpy().flatten()

def theoretical_force(x_values_mm):
    """计算理论力"""
    # 将毫米转换为米
    x_values_m = x_values_mm / 1000.0
    x_tensor = torch.tensor(x_values_m, dtype=torch.float32).unsqueeze(1)
    force = theoretical_arf(x_tensor)
    return force.numpy().flatten()

def plot_comparison():
    """绘制模型预测与理论值的对比图"""
    # 加载模型和参数
    model, norm_params = load_model_and_params()
    
    # 直接使用毫米范围，不依赖归一化参数
    x_min_mm = -25.5  # 毫米
    x_max_mm = 25.5   # 毫米
    
    # 强制设置范围，确保不被其他代码覆盖
    print(f"强制设置范围: [{x_min_mm}, {x_max_mm}] mm")
    
    print(f"绘图范围: [{x_min_mm:.1f}, {x_max_mm:.1f}] mm")
    
    # 生成测试数据点（直接使用毫米）
    x_range_mm = np.linspace(x_min_mm, x_max_mm, 1000)
    
    # 调试信息
    print(f"x_range_mm 最小值: {np.min(x_range_mm):.1f} mm")
    print(f"x_range_mm 最大值: {np.max(x_range_mm):.1f} mm")
    print(f"x_range_mm 范围: {np.max(x_range_mm) - np.min(x_range_mm):.1f} mm")
    
    # 计算理论力
    print("计算理论力...")
    theoretical_forces = theoretical_force(x_range_mm)
    
    # 计算模型预测力
    print("计算模型预测力...")
    predicted_forces = predict_force(model, norm_params, x_range_mm)
    
    # 计算相对误差
    relative_errors = np.abs(predicted_forces - theoretical_forces) / (np.abs(theoretical_forces) + 1e-30)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：力的大小对比
    ax1.plot(x_range_mm, theoretical_forces * 1e12, 'b-', label='Theoretical', linewidth=2)
    ax1.plot(x_range_mm, predicted_forces * 1e12, 'r--', label='PINN Prediction', linewidth=2)
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Acoustic Radiation Force F (pN)')
    ax1.set_title('PINN Model Prediction vs Theoretical Values (Fourier Features Enhanced)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # 强制设置坐标轴范围
    ax1.set_xlim(x_min_mm, x_max_mm)
    print(f"设置坐标轴范围: [{x_min_mm}, {x_max_mm}] mm")
    
    # 验证坐标轴范围
    xlim = ax1.get_xlim()
    print(f"实际坐标轴范围: [{xlim[0]:.1f}, {xlim[1]:.1f}] mm")
    
    # 下图：相对误差
    ax2.semilogy(x_range_mm, relative_errors * 100, 'g-', linewidth=2)
    ax2.set_xlabel('Position x (mm)')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title('PINN Prediction Relative Error')
    ax2.grid(True, alpha=0.3)
    # 强制设置坐标轴范围
    ax2.set_xlim(x_min_mm, x_max_mm)
    ax2.set_ylim(1e-3, 1e2)
    print(f"设置误差图坐标轴范围: [{x_min_mm}, {x_max_mm}] mm")
    
    plt.tight_layout()
    plt.savefig('arf_pinn_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== Statistics ===")
    print(f"Mean Relative Error: {np.mean(relative_errors) * 100:.2f}%")
    print(f"Max Relative Error: {np.max(relative_errors) * 100:.2f}%")
    print(f"Min Relative Error: {np.min(relative_errors) * 100:.2f}%")
    print(f"Error Standard Deviation: {np.std(relative_errors) * 100:.2f}%")
    
    # 打印一些关键点的对比
    print("\n=== Key Points Comparison ===")
    # 选择一些代表性的点（直接使用毫米）
    key_points_mm = [x_min_mm, x_min_mm/2, 0.0, x_max_mm/2, x_max_mm]
    print(f"Key Points Range: [{x_min_mm}, {x_max_mm}] mm")
    print("Position(mm) | Theoretical(pN) | Predicted(pN) | Rel.Error(%)")
    print("-" * 60)
    for x_mm in key_points_mm:
        idx = np.argmin(np.abs(x_range_mm - x_mm))
        theo = theoretical_forces[idx] * 1e12
        pred = predicted_forces[idx] * 1e12
        rel_err = relative_errors[idx] * 100
        print(f"{x_mm:8.1f} | {theo:10.2e} | {pred:10.2e} | {rel_err:8.2f}")

if __name__ == "__main__":
    try:
        plot_comparison()
        print("\nComparison plot saved as 'arf_pinn_comparison.png'")
    except FileNotFoundError as e:
        print(f"Error: File not found {e}")
        print("Please ensure the trained model files exist:")
        print("- arf_pinn_model.pth")
        print("- arf_pinn_normalization_params.json")
    except Exception as e:
        print(f"Error occurred: {e}")