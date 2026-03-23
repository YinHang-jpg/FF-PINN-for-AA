import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# 兼容导入 ARF.py 中的常量
try:
    from .ARF import p_0, rho_0, gamma, c_0, frequency
except Exception:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency


class Normalizer:
    """
    轻量级归一化器，供 main.py 安全导入使用。
    - transform: 对已知键按均值/方差标准化，否则原样返回
    - inverse: 对已知键反标准化，否则原样返回
    - load: 兼容两类JSON：
        1) 旧款: {'x_mean':..., 'x_std':..., 'fx_mean':..., ...}
        2) 精简: {'force_mu':..., 'force_sigma':...}
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        # 兼容两种格式
        if 'x_mean' in data:
            # 全量均值方差
            for k in ['x','y','t','r','fx','fy']:
                mean_key = f'{k}_mean'
                std_key = f'{k}_std'
                if mean_key in data and std_key in data:
                    self.stats[k] = (
                        torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                        torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                    )
        else:
            # 兼容时间因子/力参数键名
            mu_key = 'force_mu' if 'force_mu' in data else ('time_factor_mu' if 'time_factor_mu' in data else None)
            sigma_key = 'force_sigma' if 'force_sigma' in data else ('time_factor_sigma' if 'time_factor_sigma' in data else None)
            if mu_key and sigma_key:
                mu = torch.tensor(float(data[mu_key]), device=device, dtype=torch.float32)
                sigma = torch.tensor(max(float(data[sigma_key]), 1e-30), device=device, dtype=torch.float32)
                self.stats['fx'] = (mu, sigma)

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
    周期性感知的MLP：输入归一化时间 t_norm∈[0,1]，内部转换为物理时间秒，再生成 sin/cos(ω t)
    """
    def __init__(self, period_seconds: float):
        super().__init__()
        # 固定物理基频，避免频率漂移
        omega = 2 * np.pi * frequency
        self.register_buffer('omega', torch.tensor(omega, dtype=torch.float32))
        # 保存周期（秒）用于把 t_norm 映射回物理时间
        self.register_buffer('period_seconds', torch.tensor(period_seconds, dtype=torch.float32))

        # 主网络（大幅简化）
        self.layers = nn.Sequential(
            nn.Linear(2, 32),  # 仅一对 sin/cos 特征
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, t_norm):
        # 将归一化时间映射回物理时间(秒)
        t_seconds = t_norm * self.period_seconds
        phase = self.omega * t_seconds
        fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.layers(fourier_features)


def theoretical_time_factor(t):
    """
    理论时间因子：cos(ω t)
    """
    omega = 2 * np.pi * frequency
    return torch.cos(omega * t)


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("基于 ARF 解析公式的 PINN 训练（时间因子）…")
    print(f"device: {device}")

    # 数据生成（仅时间），覆盖一个周期
    period = 1.0 / frequency
    t_range = period
    
    N = 20000
    t = (torch.rand(N, 1, device=device) * t_range).float()  # [0, T]

    F_time = theoretical_time_factor(t).detach()

    # 特征缩放（仅输入t）
    t_min, t_max = 0.0, t_range
    t_scaled = (t - t_min) / (t_max - t_min)
    T_theory = t_scaled

    # 归一化到零均值单位方差
    time_factor_mu = F_time.mean()
    time_factor_sigma = F_time.std().clamp_min(1e-30)
    Y_theory = (F_time - time_factor_mu) / time_factor_sigma

    # 模型/优化器
    model = ARFNetT(period_seconds=period).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # 训练直到达到目标损失
    target_loss = 1e-7
    print_interval = 500
    epoch = 0
    losses = []

    print("开始训练… 目标损失:", target_loss)
    print("使用傅里叶特征增强时间周期性学习能力")
    print(f"采样范围: [{t_min:.2e}, {t_max:.2e}] s")
    print(f"包含周期数: {t_range/period:.1f} 个周期")
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(T_theory)
        loss = criterion(pred, Y_theory)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())  # 学习率调度
        losses.append(loss.item())

        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        if loss.item() < target_loss:
            print(f"✅ 达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        epoch += 1

    # 可视化
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    # 详细测试
    model.eval()
    with torch.no_grad():
        # 更密集的测试点，覆盖一个周期
        test_t = torch.linspace(0, t_range, 1000, device=device).unsqueeze(1)
        test_inp = (test_t - t_min) / (t_max - t_min)
        pred_norm = model(test_inp)
        pred_time = pred_norm * time_factor_sigma + time_factor_mu
        true_time = theoretical_time_factor(test_t)
        
        # 计算整体误差统计
        relative_errors = torch.abs(pred_time - true_time) / (torch.abs(true_time) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {mean_error*100:.2f}%")
        print(f"最大相对误差: {max_error*100:.2f}%")
        
        # 绘制时间因子曲线
        import matplotlib.pyplot as plt
        tt = (test_t * 1e6).detach().cpu().numpy().flatten()
        pred_np = pred_time.detach().cpu().numpy().flatten()
        true_np = true_time.detach().cpu().numpy().flatten()
        plt.figure(figsize=(10,5))
        plt.plot(tt, true_np, 'b-', label='理论值', linewidth=2)
        plt.plot(tt, pred_np, 'r--', label='PINN预测', linewidth=1.5)
        plt.xlabel('时间 (μs)')
        plt.ylabel('时间因子 cos(ωt)')
        plt.title('ARF 时间因子 vs 时间')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 保存模型与归一化参数
    torch.save(model.state_dict(), 'PINN/arf_model_t.pth')
    with open('PINN/arf_model_t_normalization_params.json', 'w') as f:
        json.dump({
            't_min': t_min, 't_max': t_max,
            'time_factor_mu': float(time_factor_mu.item()),
            'time_factor_sigma': float(time_factor_sigma.item())
        }, f)
    print("时间因子模型和归一化参数已保存。")


if __name__ == "__main__":
    main()
