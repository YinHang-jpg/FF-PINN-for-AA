import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

Re_FIXED = 1.0


class Normalizer:
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in ['R', 'Theta', 'Vr']:
            mean_key = f'{k}_mean'
            std_key = f'{k}_std'
            if mean_key in data and std_key in data:
                self.stats[k] = (
                    torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                    torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                )
        if 'force_mu' in data and 'force_sigma' in data:
            mu = torch.tensor(float(data['force_mu']), device=device, dtype=torch.float32)
            sigma = torch.tensor(max(float(data['force_sigma']), 1e-30), device=device, dtype=torch.float32)
            self.stats['Vr'] = (mu, sigma)

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


class WakeNetVrOseen(nn.Module):
    """
    输入: R_scaled (1维) + Theta 的傅里叶特征
    输出: Vr_Oseen (1维)
    Oseen 项包含 exp(-R*Re/4*(1-cos(Theta)))，需要更多傅里叶谐波。
    """
    def __init__(self, n_harmonics=16):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.register_buffer('harmonics',
                             torch.arange(1, n_harmonics + 1, dtype=torch.float32))
        input_dim = 1 + 2 * n_harmonics
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, R_scaled, Theta):
        phases = self.harmonics * Theta
        sin_f = torch.sin(phases)
        cos_f = torch.cos(phases)
        x = torch.cat([R_scaled, sin_f, cos_f], dim=1)
        return self.layers(x)


def theoretical_Vr_Oseen(R, Theta, Re=Re_FIXED):
    cos_t = torch.cos(Theta)
    exp_arg = (-R * Re / 4) * (1 - cos_t)
    exp_arg = torch.clamp(exp_arg, -500, 500)
    exp_term = torch.exp(exp_arg)
    R2 = R ** 2

    term1 = cos_t / (2 * R)
    term2 = (3 * R * (1 + cos_t) / 4) * exp_term
    term3 = 3 * (1 - exp_term) / Re
    return (term1 - term2 + term3) / R2


def main():
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PINN 训练: Wake Oseen 径向速度 Vr_Oseen(R, Theta)")
    print(f"device: {device}, Re = {Re_FIXED}")

    R_min, R_max = 2.0, 30.0
    Theta_min, Theta_max = 0.0, 2 * np.pi

    N = 40000
    R_raw = (torch.rand(N, 1, device=device) * (R_max - R_min) + R_min).float()
    Theta_raw = (torch.rand(N, 1, device=device) * (Theta_max - Theta_min) + Theta_min).float()

    Y_theory = theoretical_Vr_Oseen(R_raw, Theta_raw).detach()

    R_scaled = (R_raw - R_min) / (R_max - R_min)

    force_mu = Y_theory.mean()
    force_sigma = Y_theory.std().clamp_min(1e-30)
    Y_norm = (Y_theory - force_mu) / force_sigma

    model = WakeNetVrOseen(n_harmonics=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5000
    )

    target_loss = 1e-5
    print_interval = 500
    epoch = 0
    losses = []

    print(f"开始训练… 目标损失: {target_loss}")
    print(f"采样范围: R=[{R_min}, {R_max}], Theta=[0, 2pi]")

    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(R_scaled, Theta_raw)
        loss = criterion(pred, Y_norm)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())

        if epoch % print_interval == 0 or loss.item() < target_loss:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {lr:.2e}")
        if loss.item() < target_loss:
            print(f"达到目标损失 {target_loss:.1e}，在第 {epoch} 个epoch停止训练")
            break
        epoch += 1

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss - Wake Vr_Oseen')
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        n_test = 200
        test_R = torch.linspace(R_min, R_max, n_test, device=device)
        test_Theta = torch.linspace(0, 2 * np.pi, n_test, device=device)
        RR, TT = torch.meshgrid(test_R, test_Theta, indexing='ij')
        RR_flat = RR.reshape(-1, 1)
        TT_flat = TT.reshape(-1, 1)

        R_sc = (RR_flat - R_min) / (R_max - R_min)
        pred_norm = model(R_sc, TT_flat)
        pred_val = pred_norm * force_sigma + force_mu
        true_val = theoretical_Vr_Oseen(RR_flat, TT_flat)

        rel_err = torch.abs(pred_val - true_val) / (torch.abs(true_val) + 1e-30)
        print(f"\n=== 测试结果 ===")
        print(f"平均相对误差: {rel_err.mean().item()*100:.2f}%")
        print(f"最大相对误差: {rel_err.max().item()*100:.2f}%")

        pred_2d = pred_val.reshape(n_test, n_test).cpu().numpy()
        true_2d = true_val.reshape(n_test, n_test).cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(true_2d, extent=[0, 360, R_max, R_min], aspect='auto', cmap='RdBu_r')
        axes[0].set_title('Theoretical Vr_Oseen')
        axes[0].set_xlabel('Theta (deg)')
        axes[0].set_ylabel('R')
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(pred_2d, extent=[0, 360, R_max, R_min], aspect='auto', cmap='RdBu_r')
        axes[1].set_title('PINN Predicted Vr_Oseen')
        axes[1].set_xlabel('Theta (deg)')
        axes[1].set_ylabel('R')
        plt.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        plt.show()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PINN')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'wake_Vr_Oseen.pth'))
    with open(os.path.join(save_dir, 'wake_Vr_Oseen_norm.json'), 'w') as f:
        json.dump({
            'R_min': R_min, 'R_max': R_max,
            'Theta_min': Theta_min, 'Theta_max': Theta_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item()),
            'Re': Re_FIXED
        }, f)
    print("模型和归一化参数已保存至 PINN/。")


if __name__ == "__main__":
    main()
