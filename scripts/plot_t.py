import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import re

try:
    from mechanisms.ARF_PINN_t import ARFNetT as TrainedARFNetT
except Exception:
    TrainedARFNetT = None

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


def theoretical_time_factor(t_values_us):
    t_values_s = np.asarray(t_values_us, dtype=np.float32) / 1e6
    omega = 2 * np.pi * frequency
    return np.cos(omega * t_values_s)


def load_model_and_norms():
    # state dict
    try:
        state = torch.load('PINN/arf_model_t.pth', map_location='cpu')
    except FileNotFoundError:
        state = torch.load('arf_model_t.pth', map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    dims = _infer_layer_dims_from_state(state) or [2, 256, 256, 128, 64, 1]

    # norms
    try:
        with open('PINN/arf_model_t_normalization_params.json', 'r') as f:
            norms = json.load(f)
    except FileNotFoundError:
        with open('arf_model_t_normalization_params.json', 'r') as f:
            norms = json.load(f)

    model = None
    # Prefer the trained class for strict weight compatibility
    if TrainedARFNetT is not None:
        try:
            model = TrainedARFNetT(period_seconds=1.0 / frequency)
            model.load_state_dict(state, strict=True)
        except Exception:
            model = None
    if model is None:
        mlp = SimpleMLP(dims)
        model = TemporalWrapper(mlp, norms['t_min'], norms['t_max'])
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, norms


def main():
    model, norms = load_model_and_norms()

    # Use saved normalization bounds so plot ranges match training/validation
    t_min_s = float(norms['t_min'])
    t_max_s = float(norms['t_max'])
    t_range_us = np.linspace(t_min_s * 1e6, t_max_s * 1e6, 1000)

    # theory
    f_theory = theoretical_time_factor(t_range_us)

    # predict
    t_s = torch.tensor((t_range_us / 1e6), dtype=torch.float32).unsqueeze(1)
    t_norm = (t_s - t_min_s) / (t_max_s - t_min_s)
    with torch.no_grad():
        # If it's the trained class, it expects normalized t directly
        pred_norm = model(t_norm)
    pred_factor = pred_norm * float(norms['time_factor_sigma']) + float(norms['time_factor_mu'])

    plt.figure(figsize=(10, 5))
    plt.plot(t_range_us, f_theory, 'b-', label='Theory', linewidth=2)
    plt.plot(t_range_us, pred_factor.detach().cpu().numpy().squeeze(1), 'r--', label='PINN Prediction', linewidth=1.5)
    plt.xlabel('Time t (μs)')
    plt.ylabel('Temporal factor cos(ωt)')
    plt.title('F(t): Theory vs PINN')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Mark key points consistent with ARF_PINN_t.py (assuming t_min_s==0)
    T_s = (t_max_s - t_min_s)
    if T_s > 0:
        key_points_us = [t_min_s * 1e6 + frac * T_s * 1e6 for frac in [0.0, 0.25, 0.5, 0.75, 1.0]]
        labels = ['0', 'T/4', 'T/2', '3T/4', 'T']
        for xu, lab in zip(key_points_us, labels):
            plt.axvline(xu, color='gray', linestyle=':', alpha=0.5)
            plt.text(xu, 0.85, lab, ha='center', fontsize=8)
    plt.xlim(t_min_s * 1e6, t_max_s * 1e6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


