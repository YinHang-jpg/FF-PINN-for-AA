import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

# 兼容导入 ARF.py 中的常量与解析函数
try:
    from .ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
except Exception:
    try:
        from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
    except Exception:
        from ARF import p_0, rho_0, gamma, c_0, frequency, amplitude


class Normalizer:
    """
    轻量级归一化器，供 main.py 安全导入使用。
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        # 兼容两种格式
        if 't_mean' in data:
            # 全量均值方差
            for k in ['t', 'time_factor']:
                mean_key = f'{k}_mean'
                std_key = f'{k}_std'
                if mean_key in data and std_key in data:
                    self.stats[k] = (
                        torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                        torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                    )
        else:
            # 仅时间因子的均值方差
            if 'time_factor_mu' in data and 'time_factor_sigma' in data:
                mu = torch.tensor(float(data['time_factor_mu']), device=device, dtype=torch.float32)
                sigma = torch.tensor(max(float(data['time_factor_sigma']), 1e-30), device=device, dtype=torch.float32)
                self.stats['time_factor'] = (mu, sigma)

    def transform(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = (v - mu) / sigma
            else:
                out[k] = v
        return out

    def inverse(self, data_dict):
        out = {}
        for k, v in data_dict.items():
            if k in self.stats:
                mu, sigma = self.stats[k]
                out[k] = v * sigma + mu
            else:
                out[k] = v
        return out


class ARFNetT(nn.Module):
    """
    时间因子网络：输入 [t]，输出 [时间因子]
    使用傅里叶特征来更好地捕捉声波的时间周期性特征
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算角频率用于傅里叶特征
        omega = 2 * np.pi * frequency  # 约 2.51e6 rad/s
        
        # 傅里叶特征权重（多个频率分量）
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * omega)
        
        # 主网络（与空间模型保持一致）
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2, 256),  # sin和cos特征
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        # 生成傅里叶特征
        fourier_t = self.fourier_weights * t
        fourier_features = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        
        # 通过主网络
        return self.layers(fourier_features)


def theoretical_time_factor(t):
    """
    计算理论时间因子：cos(ωt)
    其中 ω = 2πf，f 是声波频率
    """
    omega = 2 * np.pi * frequency
    return torch.cos(omega * t)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("基于 ARF 解析公式的 PINN 训练（时间因子）…")
    print(f"device: {device}")

    # 数据生成（仅使用时间t）
    # 仅覆盖一个声波周期
    period = 1.0 / frequency  # 声波周期
    t_range = period  # 仅覆盖1个周期
    
    N = 20000  # 增加数据量，与空间模型一致
    t = (torch.rand(N, 1, device=device) * t_range).float()  # [0, T]
    
    # 理论时间因子
    time_factor_theory = theoretical_time_factor(t).detach()

    # 特征缩放（仅输入t）
    t_min, t_max = 0.0, t_range
    t_scaled = (t - t_min) / (t_max - t_min)
    T_theory = t_scaled

    # 时间因子归一化
    time_factor_mu = time_factor_theory.mean()
    time_factor_sigma = time_factor_theory.std().clamp_min(1e-30)
    Y_theory = (time_factor_theory - time_factor_mu) / time_factor_sigma

    # 模型/优化器（与空间模型保持一致）
    model = ARFNetT(fourier_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 降低学习率，与空间模型一致
    criterion = nn.MSELoss()
    
    # 学习率调度器（与空间模型一致）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # 训练直到达到目标损失
    target_loss = 1e-2  # 与空间模型一致的目标
    print_interval = 500
    epoch = 0
    losses = []

    print("开始训练… 目标损失:", target_loss)
    print("使用傅里叶特征增强时间周期性学习能力")
    print(f"采样范围: [{t_min:.2e}, {t_max:.2e}] s")
    print(f"包含周期数: {t_range/period:.1f} 个声波周期")
    print(f"声波频率: {frequency:.1e} Hz")
    print(f"声波周期: {period:.2e} s")
    print(f"训练时间范围: [0, {period*1e6:.1f}] μs")
    
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(T_theory)
        loss = criterion(pred, Y_theory)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())  # 学习率调度，与空间模型一致
        losses.append(loss.item())

        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        if loss.item() < target_loss:
            print(f"✅ 达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        epoch += 1

    # 可视化训练损失（与空间模型保持一致）
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Time Factor)')
    plt.grid(True)
    plt.show()
    
    # 详细测试和可视化
    model.eval()
    with torch.no_grad():
        # 更密集的测试点，覆盖完整范围
        test_t = torch.linspace(0, t_range, 1000, device=device).unsqueeze(1)
        test_inp = (test_t - t_min) / (t_max - t_min)
        pred_norm = model(test_inp)
        pred_time_factor = pred_norm * time_factor_sigma + time_factor_mu
        true_time_factor = theoretical_time_factor(test_t)
        
        # 计算整体误差统计
        relative_errors = torch.abs(pred_time_factor - true_time_factor) / (torch.abs(true_time_factor) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {mean_error*100:.2f}%")
        print(f"最大相对误差: {max_error*100:.2f}%")
        
        # 绘制ARF与时间的关系
        test_t_np = test_t.cpu().numpy().flatten()
        pred_factor_np = pred_time_factor.cpu().numpy().flatten()
        true_factor_np = true_time_factor.cpu().numpy().flatten()
        
        plt.figure(figsize=(10, 5))
        plt.plot(test_t_np * 1e6, true_factor_np, 'b-', label='理论值', linewidth=2)
        plt.plot(test_t_np * 1e6, pred_factor_np, 'r--', label='PINN预测', linewidth=1.5)
        plt.xlabel('时间 (μs)')
        plt.ylabel('时间因子 cos(ωt)')
        plt.title('ARF时间因子 vs 时间')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加周期标记（仅标记一个周期内的关键点）
        key_times = [0, period/4, period/2, 3*period/4, period]
        for i, t_val in enumerate(key_times):
            t_us = t_val * 1e6
            plt.axvline(t_us, color='gray', linestyle=':', alpha=0.5)
            labels = ['0', 'T/4', 'T/2', '3T/4', 'T']
            plt.text(t_us, 0.8, labels[i], ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    # 保存模型与归一化参数
    torch.save(model.state_dict(), 'arf_model_t.pth')
    with open('arf_model_t_normalization_params.json', 'w') as f:
        json.dump({
            't_min': t_min, 't_max': t_max,
            'time_factor_mu': float(time_factor_mu.item()),
            'time_factor_sigma': float(time_factor_sigma.item())
        }, f)
    print("时间因子模型和归一化参数已保存。")
    
    # 打印一些关键时间点的值
    print("\n=== 关键时间点验证 ===")
    print("时间(μs) | 理论值 | PINN预测 | 相对误差")
    print("-" * 50)
    key_times = [0, period/4, period/2, 3*period/4, period]
    labels = ['0', 'T/4', 'T/2', '3T/4', 'T']
    for i, t_val in enumerate(key_times):
        t_tensor = torch.tensor([[t_val]], device=device)
        t_scaled = (t_tensor - t_min) / (t_max - t_min)
        pred_norm = model(t_scaled)
        pred_val = pred_norm * time_factor_sigma + time_factor_mu
        true_val = theoretical_time_factor(t_tensor)
        rel_err = abs(pred_val.item() - true_val.item()) / (abs(true_val.item()) + 1e-30)
        print(f"{t_val*1e6:8.1f} | {true_val.item():7.4f} | {pred_val.item():9.4f} | {rel_err:8.1%}")


if __name__ == "__main__":
    main()
