import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# Import shared constants from ARF.py
try:
    from .ARF import p_0, rho_0, gamma, c_0, frequency
except Exception:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency


class Normalizer:
    """
    Lightweight normalizer for driver scripts.
    - transform: z-score known keys; pass through unknown keys
    - inverse: invert z-score for known keys; pass through otherwise
    - load: supports two JSON layouts:
        1) Legacy: {'x_mean':..., 'x_std':..., 'fx_mean':..., ...}
        2) Compact: {'force_mu':..., 'force_sigma':...}
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        # Support both JSON layouts
        if 'x_mean' in data:
            # full mean/std per field
            for k in ['x','y','t','r','fx','fy']:
                mean_key = f'{k}_mean'
                std_key = f'{k}_std'
                if mean_key in data and std_key in data:
                    self.stats[k] = (
                        torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                        torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                    )
        else:
            # Accept legacy key names for temporal factor / force stats
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
    MLP on normalized time in [0,1]; maps to physical seconds for sin/cos(omega t)
    """
    def __init__(self, period_seconds: float):
        super().__init__()
        # Physical carrier frequency fixed
        omega = 2 * np.pi * frequency
        self.register_buffer('omega', torch.tensor(omega, dtype=torch.float32))
        # Period in seconds for denormalizing t_norm
        self.register_buffer('period_seconds', torch.tensor(period_seconds, dtype=torch.float32))

        # Compact MLP
        self.layers = nn.Sequential(
            nn.Linear(2, 32),  # one sin/cos pair
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, t_norm):
        # Map normalized time back to seconds
        t_seconds = t_norm * self.period_seconds
        phase = self.omega * t_seconds
        fourier_features = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.layers(fourier_features)


def theoretical_time_factor(t):
    """
    Real physical time factor of the SPGF / ARF: cos(omega t).
    This is the actual carrier-time modulation (dimensionless, in [-1, 1] by
    nature of cos). The full ARF in Newtons is reconstructed downstream as
    ARF_PINN_x(x) * theoretical_time_factor(t), so this branch must NOT be
    rescaled to a different numeric range.
    """
    omega = 2 * np.pi * frequency
    return torch.cos(omega * t)


def main():
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, TensorDataset
    plt.rcParams.update({
        "figure.figsize": (7, 4),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training ARF temporal-factor PINN")
    print(f"device: {device}")

    # Time samples over one period
    period = 1.0 / frequency
    t_range = period
    
    N = 20000
    t = (torch.rand(N, 1, device=device) * t_range).float()  # [0, T]

    F_time = theoretical_time_factor(t).detach()

    # Scale time input
    t_min, t_max = 0.0, t_range
    t_scaled = (t - t_min) / (t_max - t_min)
    T_theory = t_scaled

    # Z-score targets
    time_factor_mu = F_time.mean()
    time_factor_sigma = F_time.std().clamp_min(1e-30)
    Y_theory = (F_time - time_factor_mu) / time_factor_sigma

    # Model + optimizer
    model = ARFNetT(period_seconds=period).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # Train until target loss
    target_loss = 1e-7
    print_interval = 500
    epoch = 0
    losses = []

    print("Training, target loss:", target_loss)
    print("Temporal Fourier features enabled")
    print(f"Time window [{t_min:.2e}, {t_max:.2e}] s")
    print(f"Spans {t_range/period:.1f} acoustic periods")
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(T_theory)
        loss = criterion(pred, Y_theory)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())

        if epoch % print_interval == 0 or loss.item() < target_loss:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:05d} | Loss = {loss.item():.3e} | LR = {current_lr:.2e}")
        if loss.item() < target_loss:
            print(f"Target loss {target_loss:.1e} at epoch {epoch}")
            break
        epoch += 1

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SPGF Time-Factor PINN Training Loss')
    plt.grid(True)
    plt.show()
    
    # Error report
    model.eval()
    with torch.no_grad():
        # Dense temporal test grid
        test_t = torch.linspace(0, t_range, 1000, device=device).unsqueeze(1)
        test_inp = (test_t - t_min) / (t_max - t_min)
        pred_norm = model(test_inp)
        pred_time = pred_norm * time_factor_sigma + time_factor_mu
        true_time = theoretical_time_factor(test_t)
        
        # Error statistics
        relative_errors = torch.abs(pred_time - true_time) / (torch.abs(true_time) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== Test metrics ===")
        print(f"Mean rel. error: {mean_error*100:.2f}%")
        print(f"Max rel. error: {max_error*100:.2f}%")
        print(f"Peak |cos(omega t)| (theory): {true_time.abs().max().item():.3e}")

        # Display only: normalize by peak |cos| for a consistent [-1, 1] figure scale.
        y_scale = float(true_time.abs().max().item()) + 1e-30
        import matplotlib.pyplot as plt
        tt = (test_t * 1e6).detach().cpu().numpy().flatten()
        pred_np = (pred_time / y_scale).detach().cpu().numpy().flatten()
        true_np = (true_time / y_scale).detach().cpu().numpy().flatten()
        plt.figure(figsize=(7, 4))
        plt.plot(tt, true_np, 'b-', label='Theory (normalized)', linewidth=2)
        plt.plot(tt, pred_np, 'r--', label='PINN prediction (normalized)', linewidth=1.5)
        plt.xlabel('Time (μs)')
        plt.ylabel(f'cos(ωt) / |cos|_max  (|cos|_max = {y_scale:.3e})')
        plt.title('SPGF time factor vs time (display normalized)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.show()

    # Save weights + JSON
    torch.save(model.state_dict(), 'PINN/arf_model_t.pth')
    with open('PINN/arf_model_t_normalization_params.json', 'w') as f:
        json.dump({
            't_min': t_min, 't_max': t_max,
            'time_factor_mu': float(time_factor_mu.item()),
            'time_factor_sigma': float(time_factor_sigma.item()),
            # Aliases for loaders that expect force_mu / force_sigma (e.g. UnifiedNormalizer)
            'force_mu': float(time_factor_mu.item()),
            'force_sigma': float(time_factor_sigma.item()),
        }, f)
    print("Saved ARF-t model + JSON.")


if __name__ == "__main__":
    main()
