import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

# Import constants / reference formulas from Stokes_drag.py
try:
    from .Stokes_drag import compute_air_velocity_due_to_sound
except Exception:
    from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound

# Frequency follows project config so freq_sweep_with_training stays consistent
try:
    from initialization.sound_source_standing import frequency
except Exception:
    frequency = 10000  # Hz fallback when not run from repo root
sound_pressure_level = 168.5  # dB
sound_speed = 340  # m/s
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa
rho_0 = 1.225  # kg/m³
c_0 = 340  # m/s


class Normalizer:
    """
    Lightweight normalizer for driver scripts.
    """
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in ['x', 'vx', 'vy', 't', 'fx', 'fy']:
            mean_key = f'{k}_mean'
            std_key = f'{k}_std'
            if mean_key in data and std_key in data:
                self.stats[k] = (
                    torch.tensor(data[mean_key], device=device, dtype=torch.float32),
                    torch.tensor(max(float(data[std_key]), 1e-30), device=device, dtype=torch.float32)
                )

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


class StokesNetX(nn.Module):
    """
    Fourier MLP inputs [x, vx, vy, t] -> [Fx, Fy] (Fx used)
    Fourier features capture acoustic periodicity
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
        
        # Wavenumber for Fourier features
        wavelength = sound_speed / frequency  # 0.034 m
        k = 2 * np.pi / wavelength
        
        # Random Fourier feature weights
        self.register_buffer('fourier_weights', torch.randn(1, fourier_features) * k)
        
        # Compact MLP
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 2 + 1, 32),  # sin/cos + normalized x (checkpoint layout)
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # Fx
        )

    def forward(self, x):
        # Build Fourier features from x
        fourier_x = self.fourier_weights * x
        fourier_features = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        # Concatenate normalized x (matches trained graph)
        combined_input = torch.cat([fourier_features, x], dim=1)
        
        # MLP forward
        return self.layers(combined_input)


def theoretical_stokes_force_x(x, particle_radius=1e-6):
    """
    Reference spatial factor cos(kx) from Stokes_drag.py:
    Returns cos(2*pi*x/lambda); other coefficients applied upstream
    """
    # Wavelength / wavenumber
    wavelength = sound_speed / frequency
    k = 2 * np.pi / wavelength
    
    # Spatial factor cos(2*pi*x/lambda)
    position_factor = torch.cos(k * x)
    
    return position_factor  # dimensionless in [-1, 1]


def main():
    import matplotlib.pyplot as plt
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
    print("Training Stokes spatial-factor PINN")
    print(f"device: {device}")
    print(f"Carrier frequency: {frequency} Hz")

    # Training samples along x
    wavelength = sound_speed / frequency  # 0.034 m
    x_range = 3 * wavelength
    
    N = 20000
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()  # [-0.051, 0.051] m

    F_theory = theoretical_stokes_force_x(x).detach()

    # Feature scaling (x)
    x_min, x_max = -x_range/2, x_range/2
    x_scaled = (x - x_min) / (x_max - x_min)
    X_theory = x_scaled

    # Normalize targets (Fx)
    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    Y_theory = (F_theory - force_mu) / force_sigma

    # Model + optimizer
    model = StokesNetX(fourier_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000)

    # Train until target loss
    target_loss = 1e-8
    print_interval = 500
    epoch = 0
    losses = []

    print("Training, target loss:", target_loss)
    print("Fourier features enabled")
    print(f"Sample x in [{x_min:.3f}, {x_max:.3f}] m")
    
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
            print(f"Target loss {target_loss:.1e} reached at epoch {epoch}")
            break
        epoch += 1

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Stokes Spatial-Dependence PINN Training Loss')
    plt.grid(True)
    plt.show()
    
    # Error report
    model.eval()
    with torch.no_grad():
        # Test grid
        test_x = torch.linspace(x_min, x_max, 1000, device=device).unsqueeze(1)
        test_x_scaled = (test_x - x_min) / (x_max - x_min)
        
        pred_norm = model(test_x_scaled)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_stokes_force_x(test_x)
        
        # Error statistics
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== Test metrics ===")
        print(f"Mean rel. error: {mean_error*100:.2f}%")
        print(f"Max rel. error: {max_error*100:.2f}%")

        # Plot analytic vs model along x
        x_mm = (test_x * 1000.0).detach().cpu().numpy().flatten()
        pred_fx = pred_force.detach().cpu().numpy().flatten()
        true_fx = true_force.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(7, 4))
        plt.plot(x_mm, true_fx, 'b-', label='Theory', linewidth=2)
        plt.plot(x_mm, pred_fx, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Position x (mm)')
        plt.ylabel('Position factor cos(kx) (dimensionless)')
        plt.title('Stokes position factor: theory vs PINN')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.show()

    # Save weights + JSON
    torch.save(model.state_dict(), 'PINN/stokes_model_x.pth')
    with open('PINN/stokes_model_x_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("Saved Stokes-x model and normalization JSON.")


if __name__ == "__main__":
    main()
