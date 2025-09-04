import json
import os
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_model_and_norm(
    model_path: str = "arf_pinn_model.pth",
    norm_path: str = "arf_pinn_normalization_params.json",
    device: torch.device | None = None,
):
    """
    Load trained ARF PINN model and its normalization parameters.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Local import to avoid heavy dependencies for other modules
    from mechanisms.ARF_PINN import ARFNet

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)

    model = ARFNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Required keys from training
    x_min = float(norm.get("x_min", -1.0))
    x_max = float(norm.get("x_max", 1.0))
    t_min = float(norm.get("t_min", 0.0))
    t_max = float(norm.get("t_max", 1e-3))
    force_mu = float(norm["force_mu"])  # must exist
    force_sigma = float(norm["force_sigma"])  # must exist

    return model, (x_min, x_max, t_min, t_max, force_mu, force_sigma), device


@torch.no_grad()
def predict_force(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    norm_params: Tuple[float, float, float, float, float, float],
    device: torch.device,
) -> torch.Tensor:
    """
    Predict denormalized force using the trained model.
    Shapes: x, t are (N, 1). Returns (N, 1).
    """
    x_min, x_max, t_min, t_max, force_mu, force_sigma = norm_params
    # Scale inputs to [0,1] ranges used in training
    x_scaled = (x - x_min) / (x_max - x_min)
    t_scaled = (t - t_min) / (t_max - t_min)
    model_inp = torch.cat((x_scaled, t_scaled), dim=1).to(device)
    pred_norm = model(model_inp)
    # Denormalize output
    pred_force = pred_norm * force_sigma + force_mu
    return pred_force


def compute_theoretical_force(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute theoretical acoustic radiation force using the analytical formula
    defined in mechanisms.ARF_PINN.theoretical_arf.
    """
    from mechanisms.ARF_PINN import theoretical_arf

    # theoretical_arf expects torch tensors and returns a torch tensor
    return theoretical_arf(x, t)


def main():
    model, norm_params, device = load_model_and_norm()

    # Grids within training ranges for fair comparison
    x_min, x_max, t_min, t_max, _, _ = norm_params

    # 1) F vs x at fixed t
    num_points = 400
    x_vals = torch.linspace(x_min, x_max, num_points, device=device).unsqueeze(1)
    t_fixed = torch.full_like(x_vals, fill_value=t_min)  # e.g., t = 0

    with torch.no_grad():
        f_pred_x = predict_force(model, x_vals, t_fixed, norm_params, device).cpu().numpy().squeeze()
        f_true_x = compute_theoretical_force(x_vals, t_fixed).cpu().numpy().squeeze()

    # 2) F vs t at fixed x
    t_vals = torch.linspace(t_min, t_max, num_points, device=device).unsqueeze(1)
    x_fixed_value = 0.0
    x_fixed = torch.full_like(t_vals, fill_value=x_fixed_value)

    with torch.no_grad():
        f_pred_t = predict_force(model, x_fixed, t_vals, norm_params, device).cpu().numpy().squeeze()
        f_true_t = compute_theoretical_force(x_fixed, t_vals).cpu().numpy().squeeze()

    # Convert axes for readability (e.g., mm for x, ms for t)
    x_mm = (x_vals.cpu().numpy().squeeze()) * 1000.0
    t_ms = (t_vals.cpu().numpy().squeeze()) * 1000.0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax0 = axes[0]
    ax0.plot(x_mm, f_true_x, label="Theory", color="tab:blue", lw=2)
    ax0.plot(x_mm, f_pred_x, label="PINN", color="tab:orange", lw=2, ls="--")
    ax0.set_title("F(x) at t = 0 ms")
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("Force F (N)")
    ax0.grid(True, ls=":", alpha=0.6)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(t_ms, f_true_t, label="Theory", color="tab:blue", lw=2)
    ax1.plot(t_ms, f_pred_t, label="PINN", color="tab:orange", lw=2, ls="--")
    ax1.set_title(f"F(t) at x = {x_fixed_value*1000.0:.1f} mm")
    ax1.set_xlabel("t (ms)")
    ax1.set_ylabel("Force F (N)")
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend()

    fig.suptitle("ARF: PINN vs Analytical Theory")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = "arf_pinn_results.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()


