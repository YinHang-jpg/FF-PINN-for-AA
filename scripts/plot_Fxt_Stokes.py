import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

# Physical constants
try:
    from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
except Exception:
    pass

# Physical parameters
frequency = 10000  # Hz
sound_speed = 340  # m/s


def load_stokes_models_and_norms():
    """Load the three Stokes factor nets + normalization JSON."""
    
    # Spatial factor F(x)
    from mechanisms.STOKES_PINN_x import StokesNetX
    state_x = torch.load('PINN/stokes_model_x.pth', map_location='cpu')
    model_x = StokesNetX(fourier_features=32)
    model_x.load_state_dict(state_x, strict=True)
    model_x.eval()
    
    with open('PINN/stokes_model_x_normalization_params.json', 'r') as f:
        norms_x = json.load(f)
    
    # Velocity factor F(v)
    from mechanisms.STOKES_PINN_v import StokesNetV
    state_v = torch.load('PINN/stokes_model_v.pth', map_location='cpu')
    model_v = StokesNetV()
    model_v.load_state_dict(state_v, strict=True)
    model_v.eval()
    
    with open('PINN/stokes_model_v_normalization_params.json', 'r') as f:
        norms_v = json.load(f)
    
    # Temporal factor F(t)
    from mechanisms.STOKES_PINN_t import StokesNetT
    state_t = torch.load('PINN/stokes_model_t.pth', map_location='cpu')
    period_s = 1.0 / frequency
    model_t = StokesNetT(period_seconds=period_s)
    model_t.load_state_dict(state_t, strict=True)
    model_t.eval()
    
    with open('PINN/stokes_model_t_normalization_params.json', 'r') as f:
        norms_t = json.load(f)
    
    return {
        'model_x': model_x, 'norms_x': norms_x,
        'model_v': model_v, 'norms_v': norms_v,
        'model_t': model_t, 'norms_t': norms_t
    }


def predict_stokes_factor(model, norms, values, key_prefix):
    """Predict one normalized Stokes branch output."""
    with torch.no_grad():
        tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
        
        # Normalize inputs per JSON
        if 'x' in key_prefix:
            val_min = norms['x_min']
            val_max = norms['x_max']
        elif 'v' in key_prefix:
            val_min = norms['vx_min']
            val_max = norms['vx_max']
        else:  # 't'
            val_min = norms['t_min']
            val_max = norms['t_max']
        
        scaled = (tensor - val_min) / (val_max - val_min)
        pred_norm = model(scaled)
        
        # Denormalize network output
        force = pred_norm * float(norms['force_sigma']) + float(norms['force_mu'])
        return force.squeeze(1).cpu().numpy().astype(np.float32)


def main():
    print("Loading Stokes force PINN models...")
    models_data = load_stokes_models_and_norms()
    
    # Define sampling ranges
    wavelength = sound_speed / frequency
    x_range = 3 * wavelength / 2
    x_min_m, x_max_m = -x_range/2, x_range/2
    
    v_min_ms = -0.1  # -0.1 m/s
    v_max_ms = 0.1   # 0.1 m/s
    
    t_min_s = 0.0
    t_max_s = 1.0 / frequency  # One period
    
    # Number of sampling points
    num_points = 100
    xs_m = np.linspace(x_min_m, x_max_m, num_points)
    vs_ms = np.linspace(v_min_ms, v_max_ms, num_points)
    ts_s = np.linspace(t_min_s, t_max_s, num_points)
    
    # Predict three factors
    print("Predicting spatial factor F(x)...")
    F_x = predict_stokes_factor(models_data['model_x'], models_data['norms_x'], xs_m, 'x')
    
    print("Predicting velocity factor F(v)...")
    F_v = predict_stokes_factor(models_data['model_v'], models_data['norms_v'], vs_ms, 'v')
    
    print("Predicting temporal factor F(t)...")
    F_t = predict_stokes_factor(models_data['model_t'], models_data['norms_t'], ts_s, 't')
    
    # Create 1 row x 3 columns layout with increased horizontal spacing
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.subplots_adjust(wspace=0.35)  # Increase horizontal spacing
    
    # ========== Column 1: F(x,t) at v=0.05 m/s ==========
    v_idx = int(num_points * 0.75)  # v = 0.05 m/s (75% of range from -0.1 to 0.1)
    ax = axes[0]
    
    # F(x,t,v) = F(x) * F(t) * F(v)
    F_xt = np.outer(F_t, F_x) * F_v[v_idx]
    
    # Create grid
    X_mm = xs_m * 1000.0
    T_us = ts_s * 1e6
    X_grid, T_grid = np.meshgrid(X_mm, T_us)
    
    # Plot contour
    cs = ax.contourf(X_grid, T_grid, F_xt * 1e12, levels=20, cmap='RdBu_r')
    cbar = plt.colorbar(cs, ax=ax, label='F (pN)')
    ax.set_xlabel('Position x (mm)', fontsize=12)
    ax.set_ylabel('Time t (μs)', fontsize=12)
    ax.set_title(f'F(x,t) at v={vs_ms[v_idx]:.3f} m/s', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ========== Column 2: F(x,v) at t=25μs (25% of period) ==========
    t_idx = num_points // 4  # t = T/4 (25% of period)
    ax = axes[1]
    
    # F(x,v,t) = F(x) * F(v) * F(t)
    F_xv = np.outer(F_v, F_x) * F_t[t_idx]
    
    # Create grid
    V_mms = vs_ms * 1000.0
    X_grid, V_grid = np.meshgrid(X_mm, V_mms)
    
    # Plot contour
    cs = ax.contourf(X_grid, V_grid, F_xv * 1e12, levels=20, cmap='RdBu_r')
    cbar = plt.colorbar(cs, ax=ax, label='F (pN)')
    ax.set_xlabel('Position x (mm)', fontsize=12)
    ax.set_ylabel('Velocity v (mm/s)', fontsize=12)
    ax.set_title(f'F(x,v) at t={ts_s[t_idx]*1e6:.1f} μs', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # ========== Column 3: F(v,t) at x=8.5mm (25% of range) ==========
    x_idx = int(num_points * 0.75)  # x = 8.5 mm (25% from center to max)
    ax = axes[2]
    
    # F(v,t,x) = F(v) * F(t) * F(x)
    F_vt = np.outer(F_t, F_v) * F_x[x_idx]
    
    # Create grid
    V_grid, T_grid = np.meshgrid(V_mms, T_us)
    
    # Plot contour
    cs = ax.contourf(V_grid, T_grid, F_vt * 1e12, levels=20, cmap='RdBu_r')
    cbar = plt.colorbar(cs, ax=ax, label='F (pN)')
    ax.set_xlabel('Velocity v (mm/s)', fontsize=12)
    ax.set_ylabel('Time t (μs)', fontsize=12)
    ax.set_title(f'F(v,t) at x={xs_m[x_idx]*1000:.1f} mm', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Main title
    fig.suptitle('Stokes Drag Force F(x,v,t) Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = 'scripts/stokes_force_3d_visualization.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()
