import numpy as np
import torch
import torch.nn as nn
import json
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, get_amplitude
except Exception:
    from ARF import p_0, rho_0, gamma, c_0, frequency, get_amplitude


class SimpleMLP(nn.Module):
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SpatialWrapper(nn.Module):
    def __init__(self, mlp, fourier_features: int):
        super().__init__()
        self.mlp = mlp
        wavelength = c_0 / frequency
        self.register_buffer('k', torch.tensor(2 * np.pi / wavelength, dtype=torch.float32))
        # Input dim aligned with training: 2*fourier_features
        self.register_buffer('fourier_weights', torch.randn(1, int(fourier_features), dtype=torch.float32))

    def forward(self, x_meters):
        # Broadcast phases to shape [N, F]
        phase = self.k * x_meters @ torch.ones((1, self.fourier_weights.shape[1]), device=x_meters.device) * self.fourier_weights
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.mlp(feats)


class TemporalWrapper(nn.Module):
    def __init__(self, mlp, t_min_s, t_max_s):
        super().__init__()
        self.mlp = mlp
        self.register_buffer('t_min', torch.tensor(float(t_min_s), dtype=torch.float32))
        self.register_buffer('t_max', torch.tensor(float(t_max_s), dtype=torch.float32))
        self.register_buffer('omega', torch.tensor(2 * np.pi * frequency, dtype=torch.float32))

    def forward(self, t_norm):
        t_sec = t_norm * (self.t_max - self.t_min) + self.t_min
        phase = self.omega * t_sec
        feats = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
        return self.mlp(feats)


def _infer_layer_dims_from_state(state_dict):
    linear_keys = sorted([k for k in state_dict.keys() if re.match(r"layers\.[0-9]+\.weight", k)])
    if not linear_keys:
        return None
    dims = []
    for i, k in enumerate(linear_keys):
        W = state_dict[k]
        out_dim, in_dim = W.shape
        if i == 0:
            dims.append(in_dim)
        dims.append(out_dim)
    return dims


def load_x_model_and_norms():
    # Prefer checkpoint NN class from PINN/
    from mechanisms.ARF_PINN_x import ARFNet as TrainedARFNetX
    state = torch.load('PINN/arf_model_x.pth', map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    model = TrainedARFNetX(fourier_features=32)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load normalization if present; else infer range from theory
    try:
        with open('PINN/arf_model_x_normalization_params.json', 'r') as f:
            norms = json.load(f)
    except FileNotFoundError:
        wavelength = c_0 / frequency
        half_period = wavelength / 2.0
        x_min_m = -1.5 * half_period
        x_max_m =  1.5 * half_period
        xs = np.linspace(x_min_m, x_max_m, 2000, dtype=np.float32)
        # Theoretical force shapes mu/sigma fallback
        k = 2 * np.pi / wavelength
        d_p = 2.0 * 1e-6
        offset = np.sqrt(2.0) * d_p / 4.0
        amp = get_amplitude()
        p_front = 2.0 * np.pi * amp * p_0 * gamma * np.sin(k * (xs - offset)) / wavelength
        p_back  = 2.0 * np.pi * amp * p_0 * gamma * np.sin(k * (xs + offset)) / wavelength
        force   = np.pi * d_p**2 * (p_front - p_back) / 4.0
        norms = {
            'x_min': float(x_min_m),
            'x_max': float(x_max_m),
            'force_mu': float(np.mean(force)),
            'force_sigma': float(max(np.std(force), 1e-30)),
        }
    return model, norms


def load_t_model_and_norms():
    from mechanisms.ARF_PINN_t import ARFNetT as TrainedARFNetT
    state = torch.load('PINN/arf_model_t.pth', map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    # Time normalization: read JSON if present, else one-period cos heuristic
    try:
        with open('PINN/arf_model_t_normalization_params.json', 'r') as f:
            norms = json.load(f)
    except FileNotFoundError:
        t_min_s = 0.0
        t_max_s = 1.0 / frequency
        ts = np.linspace(t_min_s, t_max_s, 2000, dtype=np.float32)
        cosv = np.cos(2 * np.pi * frequency * ts)
        norms = {
            't_min': float(t_min_s),
            't_max': float(t_max_s),
            'time_factor_mu': float(np.mean(cosv)),
            'time_factor_sigma': float(max(np.std(cosv), 1e-30)),
        }

    model = TrainedARFNetT(period_seconds=1.0 / frequency)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, norms


def main():
    model_x, norms_x = load_x_model_and_norms()
    model_t, norms_t = load_t_model_and_norms()

    # Build ranges from training-normalization ranges to ensure consistency
    x_min_m = float(norms_x['x_min'])
    x_max_m = float(norms_x['x_max'])
    t_min_s = float(norms_t['t_min'])
    t_max_s = float(norms_t['t_max'])

    num_x = 200
    num_t = 200
    xs_m = np.linspace(x_min_m, x_max_m, num_x, dtype=np.float32)
    ts_s = np.linspace(t_min_s, t_max_s, num_t, dtype=np.float32)

    # Predict spatial factor F(x) [N]
    with torch.no_grad():
        x_tensor = torch.tensor(xs_m, dtype=torch.float32).unsqueeze(1)
        # Trained ARFNetX expects normalized x in [0,1]
        x_scaled = (x_tensor - x_min_m) / (x_max_m - x_min_m)
        pred_x_norm = model_x(x_scaled)
        F_x = pred_x_norm * float(norms_x['force_sigma']) + float(norms_x['force_mu'])
        F_x = F_x.squeeze(1).cpu().numpy().astype(np.float32)

    # Predict temporal factor cos(ωt) [-]
    with torch.no_grad():
        t_tensor = torch.tensor(ts_s).unsqueeze(1)
        t_norm = (t_tensor - t_min_s) / (t_max_s - t_min_s)
        pred_t_norm = model_t(t_norm)
        T_t = pred_t_norm * float(norms_t['time_factor_sigma']) + float(norms_t['time_factor_mu'])
        T_t = T_t.squeeze(1).cpu().numpy().astype(np.float32)

    # Compose F(x,t) = F(x)*T(t)
    F_xt = np.outer(T_t, F_x)  # shape (num_t, num_x)

    # Prepare grids for plotting
    X_mm = xs_m * 1000.0
    T_us = ts_s * 1e6
    X_grid, T_grid = np.meshgrid(X_mm, T_us)

    # Plot 3D surface (pN)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X_grid, T_grid, F_xt * 1e12, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('t (μs)')
    ax.set_zlabel('F (pN)')
    ax.set_title('PINN-predicted F(x,t)')

    # Also draw a filled contour for clarity
    ax2 = fig.add_subplot(122)
    cs = ax2.contourf(X_grid, T_grid, F_xt * 1e12, levels=30, cmap='RdBu_r')
    fig.colorbar(cs, ax=ax2, label='F (pN)')
    ax2.set_xlabel('x (mm)')
    ax2.set_ylabel('t (μs)')
    ax2.set_title('PINN-predicted F(x,t) contour')

    plt.tight_layout()
    
    # Save figure
    output_path = 'scripts/arf_force_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()


