import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt

# Physical constants / sound-field helpers
from initialization.sound_source_standing import frequency, sound_pressure_level, spl_to_pressure
from mechanisms.ARF import p_0, rho_0, gamma, c_0, get_amplitude
from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound

# Defaults for analytic labels
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa


class Normalizer:
    """Z-score normalizer backed by JSON mean/std entries."""
    def __init__(self):
        self.stats = {}

    def load(self, path, device='cpu'):
        with open(path, 'r') as f:
            data = json.load(f)
        for k in ['x', 'vx', 't', 'fx']:
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


class ARFStokesNet(nn.Module):
    """
    Single-head MLP combining Fourier(x), Fourier(t), and raw (x, vx, t) channels.
    """
    def __init__(self, fourier_features=16):
        super().__init__()
        self.fourier_features = fourier_features
        
        # Wavenumber for Fourier features
        wavelength = c_0 / frequency  # 0.034 m
        k = 2 * np.pi / wavelength
        
        # Fourier weights (spatial / temporal carriers)
        self.register_buffer('fourier_weights_x', torch.randn(1, fourier_features) * k)
        self.register_buffer('fourier_weights_t', torch.randn(1, fourier_features) * 2 * np.pi * frequency)
        
        self.layers = nn.Sequential(
            nn.Linear(fourier_features * 4 + 3, 512),  # sin/cos(x), sin/cos(t), x, vx, t
            nn.Tanh(),
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Fx
        )

    def forward(self, x, vx, t):
        fourier_x = self.fourier_weights_x * x
        fourier_features_x = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
        
        fourier_t = self.fourier_weights_t * t
        fourier_features_t = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
        
        combined_input = torch.cat([fourier_features_x, fourier_features_t, x, vx, t], dim=1)
        
        return self.layers(combined_input)


def theoretical_arf_stokes_force(x, vx, t, particle_radius=1e-6):
    """
    Analytic combined Fx used as PINN supervision:
      ARF x-projection from front/back pressures;
      Stokes drag against acoustic streaming velocity u_x(x,t).
    """
    d_p = 2 * particle_radius
    
    # Wavelength / wavenumber
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    
    sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
    A = sound_pressure / (rho_0 * c_0 * 2 * np.pi * frequency)
    
    offset = np.sqrt(2) * d_p / 4
    x_front = x - offset
    x_back = x + offset
    
    p_front = 2 * 2 * np.pi * A * p_0 * gamma * torch.sin(k * x_front) * torch.cos(omega * t) / wavelength
    p_back = 2 * 2 * np.pi * A * p_0 * gamma * torch.sin(k * x_back) * torch.cos(omega * t) / wavelength
    
    pressure_diff = p_front - p_back
    F_arf = np.pi * d_p**2 * pressure_diff / 4
    
    ux = -2 * np.pi * frequency * A * torch.cos(k * x) * torch.cos(omega * t)
    
    relative_velocity = vx - ux
    
    drag_coeff = 3.0 * np.pi * viscosity * fixed_diameter / fixed_cunningham
    F_stokes = -drag_coeff * relative_velocity
    
    F_total = F_arf + F_stokes
    
    return F_total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training unified ARF+Stokes PINN against analytic reference")
    print(f"device: {device}")

    N = 50000
    
    wavelength = c_0 / frequency
    x_range = 3 * wavelength
    x = (torch.rand(N, 1, device=device) * x_range - x_range/2).float()
    
    vx = (torch.rand(N, 1, device=device) * 0.2 - 0.1).float()
    
    period = 1.0 / frequency
    t_range = 2 * period
    t = (torch.rand(N, 1, device=device) * t_range).float()
    
    F_theory = theoretical_arf_stokes_force(x, vx, t).detach()
    
    x_min, x_max = -x_range/2, x_range/2
    vx_min, vx_max = -0.1, 0.1
    t_min, t_max = 0.0, t_range
    
    x_norm = (x - x_min) / (x_max - x_min)
    vx_norm = (vx - vx_min) / (vx_max - vx_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    force_mu = F_theory.mean()
    force_sigma = F_theory.std().clamp_min(1e-30)
    F_norm = (F_theory - force_mu) / force_sigma
    
    # Model + optimizer
    model = ARFStokesNet(fourier_features=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # LR scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000)
    
    # Train
    target_loss = 1e-2
    print_interval = 1000
    epoch = 0
    losses = []
    
    print("Training, target loss:", target_loss)
    print(f"Sampling: x in [{x_min:.3f}, {x_max:.3f}] m, vx in [{vx_min:.3f}, {vx_max:.3f}] m/s, t in [{t_min:.2e}, {t_max:.2e}] s")
    
    while True:
        model.train()
        optimizer.zero_grad()
        pred = model(x_norm, vx_norm, t_norm)
        loss = criterion(pred, F_norm)
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
    
    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ARF+Stokes Unified Model Training Loss')
    plt.grid(True)
    plt.show()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Test grid
        test_x = torch.linspace(x_min, x_max, 50, device=device).unsqueeze(1)
        test_vx = torch.linspace(vx_min, vx_max, 10, device=device).unsqueeze(1)
        test_t = torch.linspace(t_min, t_max, 20, device=device).unsqueeze(1)
        
        test_x_grid, test_vx_grid, test_t_grid = torch.meshgrid(
            test_x.squeeze(), test_vx.squeeze(), test_t.squeeze(), indexing='ij'
        )
        test_x_flat = test_x_grid.flatten().unsqueeze(1)
        test_vx_flat = test_vx_grid.flatten().unsqueeze(1)
        test_t_flat = test_t_grid.flatten().unsqueeze(1)
        
        test_x_norm = (test_x_flat - x_min) / (x_max - x_min)
        test_vx_norm = (test_vx_flat - vx_min) / (vx_max - vx_min)
        test_t_norm = (test_t_flat - t_min) / (t_max - t_min)
        
        pred_norm = model(test_x_norm, test_vx_norm, test_t_norm)
        pred_force = pred_norm * force_sigma + force_mu
        true_force = theoretical_arf_stokes_force(test_x_flat, test_vx_flat, test_t_flat)
        
        relative_errors = torch.abs(pred_force - true_force) / (torch.abs(true_force) + 1e-30)
        mean_error = torch.mean(relative_errors).item()
        max_error = torch.max(relative_errors).item()
        
        print(f"\n=== Test metrics ===")
        print(f"Mean rel. error: {mean_error*100:.2f}%")
        print(f"Max rel. error: {max_error*100:.2f}%")
        
        vx_zero = torch.zeros_like(test_x)
        t_zero = torch.zeros_like(test_x)
        vx_zero_norm = (vx_zero - vx_min) / (vx_max - vx_min)
        t_zero_norm = (t_zero - t_min) / (t_max - t_min)
        
        test_x_norm_fixed = (test_x - x_min) / (x_max - x_min)
        
        pred_fx = model(test_x_norm_fixed, vx_zero_norm, t_zero_norm)
        pred_fx_denorm = pred_fx * force_sigma + force_mu
        true_fx = theoretical_arf_stokes_force(test_x, vx_zero, t_zero)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        x_mm = (test_x * 1000).detach().cpu().numpy().flatten()
        pred_pn = (pred_fx_denorm * 1e12).detach().cpu().numpy().flatten()
        true_pn = (true_fx * 1e12).detach().cpu().numpy().flatten()
        plt.plot(x_mm, true_pn, 'b-', label='Theory', linewidth=2)
        plt.plot(x_mm, pred_pn, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Position x (mm)')
        plt.ylabel('Force Fx (pN)')
        plt.title('F(x) at vx=0, t=0')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        x_zero = torch.zeros_like(test_t)
        vx_zero_t = torch.zeros_like(test_t)
        x_zero_norm = (x_zero - x_min) / (x_max - x_min)
        vx_zero_t_norm = (vx_zero_t - vx_min) / (vx_max - vx_min)
        
        test_t_norm_fixed = (test_t - t_min) / (t_max - t_min)
        
        pred_ft = model(x_zero_norm, vx_zero_t_norm, test_t_norm_fixed)
        pred_ft_denorm = pred_ft * force_sigma + force_mu
        true_ft = theoretical_arf_stokes_force(x_zero, vx_zero_t, test_t)
        
        t_us = (test_t * 1e6).detach().cpu().numpy().flatten()
        pred_ft_pn = (pred_ft_denorm * 1e12).detach().cpu().numpy().flatten()
        true_ft_pn = (true_ft * 1e12).detach().cpu().numpy().flatten()
        plt.plot(t_us, true_ft_pn, 'b-', label='Theory', linewidth=2)
        plt.plot(t_us, pred_ft_pn, 'r--', label='PINN Prediction', linewidth=1.5)
        plt.xlabel('Time (μs)')
        plt.ylabel('Force Fx (pN)')
        plt.title('F(t) at x=0, vx=0')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Save weights + JSON
    torch.save(model.state_dict(), 'PINN/arf_stokes_unified_model.pth')
    with open('PINN/arf_stokes_unified_normalization_params.json', 'w') as f:
        json.dump({
            'x_min': x_min, 'x_max': x_max,
            'vx_min': vx_min, 'vx_max': vx_max,
            't_min': t_min, 't_max': t_max,
            'force_mu': float(force_mu.item()),
            'force_sigma': float(force_sigma.item())
        }, f)
    print("Saved unified ARF+Stokes checkpoint and normalization JSON to PINN/.")


if __name__ == "__main__":
    main()
