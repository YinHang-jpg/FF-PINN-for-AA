from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# Import shared constants from ARF.py
try:
    from .ARF import p_0, rho_0, gamma, c_0, frequency, get_amplitude
except Exception:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, get_amplitude


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
            # scalar force mean/std
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
    Fourier-feature MLP: [x] -> [Fx]
    Uses Fourier features to encode acoustic periodicity
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # Wavenumber for Fourier features
        wavelength = c_0 / frequency  # 0.034 m
        k = 2 * np.pi / wavelength
        
        # Random Fourier feature weights
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
        
        # Compact MLP
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2, 64),  # sin/cos features
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Build Fourier features
        fourier_x = self.fourier_weights * x
        fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        
        # MLP forward
        return self.layers(fourier_features)


def theoretical_arf(x, particle_radius=1e-6):
    """
    SPGF / ARF spatial force at t = 0 (Newtons), same construction as mechanisms/ARF.py
    with cos(omega t) = 1:

        F(x, 0) = π d_p^2 [p(x - √2 d_p/4, 0) - p(x + √2 d_p/4, 0)] / 4

    with p(x, 0) = (4π A p_0 γ sin(k x)) / λ.
    """
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength

    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x - offset
    x_back = x + offset

    A = torch.tensor(get_amplitude(), dtype=x.dtype, device=x.device)
    p_front = 2.0 * 2.0 * np.pi * A * p_0 * gamma * torch.sin(k * x_front) / wavelength
    p_back = 2.0 * 2.0 * np.pi * A * p_0 * gamma * torch.sin(k * x_back) / wavelength

    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p ** 2 * pressure_diff / 4.0
    return force_magnitude


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
    print("Training ARF spatial PINN")
    print(f"device: {device}")

    # Samples along x at t=0 reference
    # Wide x window
    wavelength = c_0 / frequency  # 34 mm
    period = wavelength / 2  # half-wavelength
    x_range = 3 * period
    
    N = 20000
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()  # [-25.5, 25.5] mm

    F_theory = theoretical_arf(x).detach()

    # Feature scaling (x)
    x_min, x_max = -x_range/2, x_range/2
    x_scaled = (x - x_min) / (x_max - x_min)
    X_theory = x_scaled

    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    Y_theory = (F_theory - force_mu) / force_sigma

    # Model + optimizer
    model = ARFNet(fourier_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # Train until target loss
    target_loss = 1e-5
    print_interval = 500
    epoch = 0
    losses = []

    print("Training, target loss:", target_loss)
    print("Fourier features enabled")
    print(f"x window [{x_min:.1f}, {x_max:.1f}] mm")
    print(f"Covers {x_range/period:.1f} half-wavelengths")
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(X_theory)
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
    plt.title('SPGF Spatial-Force PINN Training Loss')
    plt.grid(True)
    Path('PINN').mkdir(parents=True, exist_ok=True); plt.savefig(f'PINN/{Path(__file__).stem}_plot.png', dpi=200, bbox_inches='tight'); plt.close()
    
    # Error report
    model.eval()
    with torch.no_grad():
        # Dense test grid
        test_x = torch.linspace(x_min, x_max, 1000, device=device).unsqueeze(1)
        test_inp = (test_x - x_min) / (x_max - x_min)
        pred_norm = model(test_inp)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_arf(test_x)
        
        # Error statistics on the real physical force
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()

        print(f"\n=== Test metrics ===")
        print(f"Mean rel. error: {mean_error*100:.2f}%")
        print(f"Max rel. error: {max_error*100:.2f}%")
        print(f"Peak |F_x| (theory): {true_force.abs().max().item():.3e} N")

        # Display only: rescale theory and prediction by the theoretical peak |F|
        # so the plot stays in [-1, 1]. The trained model / JSON encode real Newtons.
        y_scale = float(true_force.abs().max().item()) + 1e-30
        x_mm = (test_x * 1000.0).detach().cpu().numpy().flatten()
        pred_pn = (pred_force / y_scale).detach().cpu().numpy().flatten()
        true_pn = (true_force / y_scale).detach().cpu().numpy().flatten()
        plt.figure(figsize=(7, 4))
        plt.plot(x_mm, true_pn, 'b-', label='Theory (normalized)', linewidth=2)
        plt.plot(x_mm, pred_pn, 'r--', label='PINN prediction (normalized)', linewidth=1.5)
        plt.xlabel('Position x (mm)')
        plt.ylabel(f'F_x / |F_x|_max  (|F_x|_max = {y_scale:.3e} N)')
        plt.title('SPGF spatial force: theory vs PINN (display normalized)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-1.1, 1.1)
        plt.xlim(x_min * 1000.0, x_max * 1000.0)
        plt.tight_layout()
        Path('PINN').mkdir(parents=True, exist_ok=True); plt.savefig(f'PINN/{Path(__file__).stem}_plot.png', dpi=200, bbox_inches='tight'); plt.close()
        print("\nPosition (mm) | Pred (N) | Theory (N) | Relative error")
        print("-"*60)
        for i in range(0, 100, 10):
            xv = float(test_x[i].item() * 1000)
            pv = float(pred_force[i].item())
            tv = float(true_force[i].item())
            rel = abs(pv - tv) / (abs(tv) + 1e-30)
            print(f"{xv:8.1f} | {pv:11.2e} | {tv:11.2e} | {rel:8.1%}")

    # Save weights + JSON under PINN/ for plotting utilities
    Path('PINN').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'PINN/arf_model_x.pth')
    with open('PINN/arf_model_x_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("Wrote weights + JSON to PINN/.")


if __name__ == "__main__":
    main()
