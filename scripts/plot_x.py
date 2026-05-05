import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import re

try:
    # Prefer the exact training class if compatible
    from mechanisms.ARF_PINN_x import ARFNet as TrainedARFNetX
except Exception:
    TrainedARFNetX = None

try:
    from mechanisms.ARF import p_0, rho_0, gamma, c_0, frequency, amplitude
except Exception:
    from ARF import p_0, rho_0, gamma, c_0, frequency, amplitude


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
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
        wavelength = c_0 / frequency
        self.register_buffer('k', torch.tensor(2 * np.pi / wavelength, dtype=torch.float32))

    def forward(self, x_meters):
        phase = self.k * x_meters
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


def theoretical_spatial_factor(x_values_mm, particle_radius=1e-6):
    x_values_m = np.asarray(x_values_mm, dtype=np.float32) / 1000.0
    wavelength = c_0 / frequency
    k = 2 * np.pi / wavelength
    d_p = 2.0 * particle_radius
    offset = np.sqrt(2.0) * d_p / 4.0
    x_front = x_values_m - offset
    x_back = x_values_m + offset
    p_front = 2.0 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_front) / wavelength
    p_back = 2.0 * np.pi * amplitude * p_0 * gamma * np.sin(k * x_back) / wavelength
    pressure_diff = p_front - p_back
    force_magnitude = np.pi * d_p**2 * pressure_diff / 4.0
    return force_magnitude


def load_model_and_norms():
    # state dict
    try:
        state = torch.load('PINN/arf_model_x.pth', map_location='cpu')
    except FileNotFoundError:
        state = torch.load('arf_model_x.pth', map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    dims = _infer_layer_dims_from_state(state) or [2, 256, 256, 128, 64, 1]
    model = None
    # Prefer the trained class for strict weight compatibility; else fallback wrapper
    if TrainedARFNetX is not None:
        try:
            model = TrainedARFNetX(fourier_features=32)
            model.load_state_dict(state, strict=True)
        except Exception:
            model = None
    if model is None:
        # Fallback to wrapper that builds sin/cos features from meters
        mlp = SimpleMLP(dims)
        model = SpatialWrapper(mlp)
        model.load_state_dict(state, strict=False)
    model.eval()

    # norms
    try:
        with open('PINN/arf_model_x_normalization_params.json', 'r') as f:
            norms = json.load(f)
    except FileNotFoundError:
        try:
            with open('arf_model_x_normalization_params.json', 'r') as f:
                norms = json.load(f)
        except FileNotFoundError:
            # try legacy combined model norms
            try:
                with open('PINN/arf_pinn_normalization_params.json', 'r') as f:
                    norms = json.load(f)
            except FileNotFoundError:
                try:
                    with open('arf_pinn_normalization_params.json', 'r') as f:
                        norms = json.load(f)
                except FileNotFoundError:
                    # fallback from theoretical span
                    wavelength = c_0 / frequency
                    half_period = wavelength / 2.0
                    x_min_m = -1.5 * half_period
                    x_max_m = 1.5 * half_period
                    xs = np.linspace(x_min_m, x_max_m, 1000, dtype=np.float32)
                    theo = theoretical_spatial_factor(xs * 1000.0)
                    norms = {
                        'x_min': float(x_min_m),
                        'x_max': float(x_max_m),
                        'force_mu': float(np.mean(theo)),
                        'force_sigma': float(max(np.std(theo), 1e-30)),
                    }
    return model, norms


def main():
    model, norms = load_model_and_norms()

    # Use saved normalization bounds so plots match training/validation ranges
    x_min_mm = float(norms['x_min']) * 1000.0
    x_max_mm = float(norms['x_max']) * 1000.0
    x_range_mm = np.linspace(x_min_mm, x_max_mm, 1000)

    # theory
    f_theory = theoretical_spatial_factor(x_range_mm)  # N

    # predict
    x_m = torch.tensor((x_range_mm / 1000.0), dtype=torch.float32).unsqueeze(1)
    # If the model is the trained ARFNetX (expects scaled input), feed x_scaled
    is_trained_class = (TrainedARFNetX is not None) and isinstance(model, TrainedARFNetX)
    with torch.no_grad():
        if is_trained_class:
            x_scaled = (x_m - float(norms['x_min'])) / (float(norms['x_max']) - float(norms['x_min']))
            pred_norm = model(x_scaled)
        else:
            pred_norm = model(x_m)  # wrapper expects meters, builds sin/cos internally
    pred_force = pred_norm * float(norms['force_sigma']) + float(norms['force_mu'])

    plt.figure(figsize=(10, 5))
    plt.plot(x_range_mm, f_theory * 1e12, 'b-', label='Theory', linewidth=2)
    plt.plot(x_range_mm, pred_force.detach().cpu().numpy().squeeze(1) * 1e12, 'r--', label='PINN Prediction', linewidth=1.5)
    plt.xlabel('Position x (mm)')
    plt.ylabel('Acoustic radiation force F (pN)')
    plt.title('F(x): Theory vs PINN')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(x_min_mm, x_max_mm)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


