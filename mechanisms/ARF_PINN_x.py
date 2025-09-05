import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

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
            # 仅力的均值方差（单输出模型时也可使用）
            if 'force_mu' in data and 'force_sigma' in data:
                mu = torch.tensor(float(data['force_mu']), device=device, dtype=torch.float32)
                sigma = torch.tensor(max(float(data['force_sigma']), 1e-30), device=device, dtype=torch.float32)
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


class ARFNet(nn.Module):
    """
    周期性感知的MLP：输入 [x]，输出 [Fx]
    使用傅里叶特征和周期性激活函数来更好地捕捉声波的周期性特征
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # 计算波数用于傅里叶特征
        wavelength = c_0 / frequency  # 0.034 m
        k = 2 * np.pi / wavelength    # 约 184.8 rad/m
        
        # 傅里叶特征权重（多个频率分量）
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
        
        # 主网络
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

    def forward(self, x):
        # 生成傅里叶特征
        fourier_x = self.fourier_weights * x
        fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        
        # 通过主网络
        return self.layers(fourier_features)


def theoretical_arf(x, particle_radius=1e-6):
    """
    基于 ARF.py 中的物理公式计算理论声辐射力（t=0）：
    F = π d_p^2 [p(x-√2 d_p/4,0) - p(x+√2 d_p/4,0)] / 4
    其中 p(x,0) = 2π A p0 γ sin(k x) / λ
    """
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength

    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x - offset
    x_back = x + offset

    # t=0时，cos(ωt) = cos(0) = 1
    p_front = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_front) / wavelength
    p_back = 2.0 * np.pi * amplitude * p_0 * gamma * torch.sin(k * x_back) / wavelength

    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4.0
    return force_magnitude


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("基于 ARF 解析公式的 PINN 训练（压力梯度力）…")
    print(f"device: {device}")

    # 数据生成（仅使用x，t=0）
    # 扩大采样范围以包含完整的声波周期
    wavelength = c_0 / frequency  # 34 mm
    period = wavelength / 2  # 17 mm (半波长)
    x_range = 3 * period  # 覆盖3个半波长 = 51 mm
    
    N = 20000  # 增加数据量
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()  # [-25.5, 25.5] mm

    F_theory = theoretical_arf(x).detach()

    # 特征缩放（仅输入x）
    x_min, x_max = -x_range/2, x_range/2
    x_scaled = (x - x_min) / (x_max - x_min)
    X_theory = x_scaled

    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    Y_theory = (F_theory - force_mu) / force_sigma

    # 模型/优化器
    model = ARFNet(fourier_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 降低学习率
    criterion = nn.MSELoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # 训练直到达到目标损失
    target_loss = 1e-5  # 更严格的目标
    print_interval = 500
    epoch = 0
    losses = []

    print("开始训练… 目标损失:", target_loss)
    print("使用傅里叶特征增强周期性学习能力")
    print(f"采样范围: [{x_min:.1f}, {x_max:.1f}] mm")
    print(f"包含周期数: {x_range/period:.1f} 个半波长")
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(X_theory)
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
        # 更密集的测试点，覆盖完整范围
        test_x = torch.linspace(x_min, x_max, 100, device=device).unsqueeze(1)
        test_inp = (test_x - x_min) / (x_max - x_min)
        pred_norm = model(test_inp)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_arf(test_x)
        
        # 计算整体误差统计
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {mean_error*100:.2f}%")
        print(f"最大相对误差: {max_error*100:.2f}%")

        # 绘制 F(x) 理论值 与 模型预测 曲线
        x_mm = (test_x * 1000.0).detach().cpu().numpy().flatten()
        pred_pn = (pred_force * 1e12).detach().cpu().numpy().flatten()
        true_pn = (true_force * 1e12).detach().cpu().numpy().flatten()
        plt.figure(figsize=(10, 5))
        plt.plot(x_mm, true_pn, 'b-', label='理论值', linewidth=2)
        plt.plot(x_mm, pred_pn, 'r--', label='PINN预测', linewidth=1.5)
        plt.xlabel('位置 x (mm)')
        plt.ylabel('声辐射力 F (pN)')
        plt.title('F(x) 理论 vs PINN')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(x_min * 1000.0, x_max * 1000.0)
        plt.tight_layout()
        plt.show()
        print("\n位置(mm) | 预测力(pN) | 理论力(pN) | 相对误差")
        print("-"*60)
        for i in range(0, 100, 10):  # 每10个点显示一个
            xv = float(test_x[i].item() * 1000)
            pv = float(pred_force[i].item() * 1e12)  # 转换为pN
            tv = float(true_force[i].item() * 1e12)  # 转换为pN
            rel = abs(pv - tv) / (abs(tv) + 1e-30)
            print(f"{xv:8.1f} | {pv:11.2e} | {tv:11.2e} | {rel:8.1%}")

    # 保存模型与归一化参数到 PINN/ 目录，供可视化脚本使用
    torch.save(model.state_dict(), 'PINN/arf_model_x.pth')
    with open('PINN/arf_model_x_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("模型和归一化参数已保存至 PINN/。")


if __name__ == "__main__":
    main()
