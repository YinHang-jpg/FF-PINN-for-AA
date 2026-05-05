import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
 
# Physical constants / helpers (sound field, Stokes air velocity)
from initialization.sound_source_standing import frequency, sound_pressure_level, spl_to_pressure
from mechanisms.ARF import p_0, rho_0, gamma, c_0, get_amplitude
from mechanisms.Stokes_drag import compute_air_velocity_due_to_sound
 
# Default fluid / particle scales for combined-force demo paths
viscosity = 1.8e-5  # Pa·s
fixed_diameter = 2e-6  # m
fixed_cunningham = 1.083
reference_pressure = 20e-6  # Pa
 
 
class UnifiedNormalizer:
    """Aggregates normalization stats from per-branch JSON files."""
    def __init__(self):
        self.stats = {}
 
    def load_from_individual_models(self, model_paths, device='cpu'):
        """Load mean/std and input ranges used at inference."""
        # ARF branches
        arf_x_norm_path = os.path.join(model_paths['arf_x_norm'])
        arf_t_norm_path = os.path.join(model_paths['arf_t_norm'])
       
        # Stokes branches
        stokes_x_norm_path = os.path.join(model_paths['stokes_x_norm'])
        stokes_t_norm_path = os.path.join(model_paths['stokes_t_norm'])
        stokes_v_norm_path = os.path.join(model_paths['stokes_v_norm'])
       
        # ARF x JSON
        if os.path.exists(arf_x_norm_path):
            with open(arf_x_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['x_min'] = params.get('x_min', 0.0)
            self.stats['x_max'] = params.get('x_max', 0.1)
            self.stats['arf_fx_mu'] = params.get('force_mu', 0.0)
            self.stats['arf_fx_sigma'] = params.get('force_sigma', 1.0)
       
        # ARF t JSON
        if os.path.exists(arf_t_norm_path):
            with open(arf_t_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['t_min'] = params.get('t_min', 0.0)
            self.stats['t_max'] = params.get('t_max', 1.0/frequency)
            self.stats['arf_ft_mu'] = params.get('force_mu', 0.0)
            self.stats['arf_ft_sigma'] = params.get('force_sigma', 1.0)
       
        # Stokes x JSON (includes x span)
        if os.path.exists(stokes_x_norm_path):
            with open(stokes_x_norm_path, 'r') as f:
                params = json.load(f)
            # Stokes-specific x window
            self.stats['stokes_x_min'] = params.get('x_min', self.stats.get('x_min', 0.0))
            self.stats['stokes_x_max'] = params.get('x_max', self.stats.get('x_max', 0.1))
            self.stats['stokes_fx_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_fx_sigma'] = params.get('force_sigma', 1.0)
       
        # Stokes t JSON (includes t span)
        if os.path.exists(stokes_t_norm_path):
            with open(stokes_t_norm_path, 'r') as f:
                params = json.load(f)
            # Stokes-specific time window
            self.stats['stokes_t_min'] = params.get('t_min', self.stats.get('t_min', 0.0))
            self.stats['stokes_t_max'] = params.get('t_max', self.stats.get('t_max', 1.0/frequency))
            self.stats['stokes_ft_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_ft_sigma'] = params.get('force_sigma', 1.0)
       
        # Stokes v JSON
        if os.path.exists(stokes_v_norm_path):
            with open(stokes_v_norm_path, 'r') as f:
                params = json.load(f)
            self.stats['vx_min'] = params.get('vx_min', -0.1)
            self.stats['vx_max'] = params.get('vx_max', 0.1)
            self.stats['stokes_fv_mu'] = params.get('force_mu', 0.0)
            self.stats['stokes_fv_sigma'] = params.get('force_sigma', 1.0)
 
    def normalize_inputs(self, x, vx, t):
        """Map raw x, vx, t to [0,1] tensors; x and t use periodic wrapping."""
        # Wrap x into the training interval
        x_min = self.stats['x_min']
        x_max = self.stats['x_max']
        x_range = x_max - x_min
        if x_range <= 0:
            raise ValueError(f"Invalid x_range in UnifiedNormalizer: x_min={x_min}, x_max={x_max}")
        x_wrapped = ((x - x_min) % x_range) + x_min
        x_norm = (x_wrapped - x_min) / x_range
       
        # Velocity channel
        vx_norm = (vx - self.stats['vx_min']) / (self.stats['vx_max'] - self.stats['vx_min'])
        vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
       
        period = self.stats['t_max'] - self.stats['t_min']  # training period
        t_norm = ((t - self.stats['t_min']) % period) / period
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
       
        return x_norm, vx_norm, t_norm
 
    def normalize_inputs_stokes(self, x, vx, t):
        """Like normalize_inputs but uses Stokes-specific x/t windows."""
        # Stokes x window
        x_min = self.stats.get('stokes_x_min', self.stats['x_min'])
        x_max = self.stats.get('stokes_x_max', self.stats['x_max'])
        x_range = x_max - x_min
        if x_range <= 0:
            raise ValueError(f"Invalid stokes_x_range in UnifiedNormalizer: x_min={x_min}, x_max={x_max}")
        # Periodic x wrap (Stokes span)
        x_wrapped = ((x - x_min) % x_range) + x_min
        x_norm = (x_wrapped - x_min) / x_range
 
        # vx uses global vx_min / vx_max
        vx_norm = (vx - self.stats['vx_min']) / (self.stats['vx_max'] - self.stats['vx_min'])
        vx_norm = torch.clamp(vx_norm, 0.0, 1.0)
 
        t_min = self.stats.get('stokes_t_min', self.stats['t_min'])
        t_max = self.stats.get('stokes_t_max', self.stats['t_max'])
        period = t_max - t_min
 
        t_norm = ((t - t_min) % period) / period
        t_norm = torch.clamp(t_norm, 0.0, 1.0)
 
        return x_norm, vx_norm, t_norm
 
    def denormalize_arf_force(self, fx_norm, ft_norm):
        """ARF spatial / temporal factors back to physical scaling."""
        fx = fx_norm * self.stats['arf_fx_sigma'] + self.stats['arf_fx_mu']
        ft = ft_norm * self.stats['arf_ft_sigma'] + self.stats['arf_ft_mu']
        return fx, ft
 
    def denormalize_stokes_force(self, fx_norm, ft_norm, fv_norm):
        """Stokes factor branches back from normalized network outputs."""
        fx = fx_norm * self.stats['stokes_fx_sigma'] + self.stats['stokes_fx_mu']
        ft = ft_norm * self.stats['stokes_ft_sigma'] + self.stats['stokes_ft_mu']
        fv = fv_norm * self.stats['stokes_fv_sigma'] + self.stats['stokes_fv_mu']
        return fx, ft, fv
 
 
class UnifiedPINN(nn.Module):
    """
    Monolithic dual-head SPGF network (ARF branch + Stokes branch).
    Inputs: x, vx, t (normalized upstream). Outputs: total Fx and branch-wise Fx.
    """
    def __init__(self, fourier_features=32):
        super().__init__()
        self.fourier_features = fourier_features
       
        # Fourier feature wavenumbers
        wavelength = c_0 / frequency
        k = 2 * np.pi / wavelength
        omega = 2 * np.pi * frequency
       
        # Spatial Fourier weights
        self.register_buffer('fourier_weights_x', torch.randn(1, fourier_features) * k)
       
        # Temporal Fourier weights
        self.register_buffer('fourier_weights_t', torch.randn(1, fourier_features) * omega)
       
        self.arf_net = nn.Sequential(
            nn.Linear(fourier_features * 4, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
       
        self.stokes_net = nn.Sequential(
            nn.Linear(fourier_features * 4, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
 
    def forward(self, x, vx, t):
        fourier_x = self.fourier_weights_x * x
        fourier_features_x = torch.cat([torch.sin(fourier_x), torch.cos(fourier_x)], dim=1)
       
        fourier_t = self.fourier_weights_t * t
        fourier_features_t = torch.cat([torch.sin(fourier_t), torch.cos(fourier_t)], dim=1)
       
        combined_input = torch.cat([fourier_features_x, fourier_features_t], dim=1)
       
        arf_fx = self.arf_net(combined_input)
        stokes_fx = self.stokes_net(combined_input)
       
        total_fx = arf_fx + stokes_fx
       
        return total_fx, arf_fx, stokes_fx
 
 
class PhysicalUnifiedModel(nn.Module):
    """Inference wrapper: five sub-networks + analytic Stokes-amplitude closure -> Fx [N]."""
    def __init__(self, models, unified_norm, device='cpu'):
        super().__init__()
        self.models = models
        self.unified_norm = unified_norm
        self.device = torch.device(device)
        
        t_max = unified_norm.stats.get('t_max', 1.0/10000)
        self.inferred_frequency = 1.0 / t_max if t_max > 0 else 10000
        print(f"  [PhysicalUnifiedModel] Inferred f = {self.inferred_frequency:.1f} Hz (t_max={t_max:.6f} s)")
        print(f"  [PhysicalUnifiedModel] Wavelength ~ {340.0/self.inferred_frequency*1000:.2f} mm")
        print(f"  [PhysicalUnifiedModel] Nodal spacing ~ {340.0/self.inferred_frequency*1000/2:.2f} mm")
 
    @torch.no_grad()
    def forward(self, x, vx, t):
        # x, vx, t: 1D Tensor (N,)
        x = x.to(self.device).float()
        vx = vx.to(self.device).float()
        t = t.to(self.device).float()
 
        # ARF factors
        x_norm, _, t_norm = self.unified_norm.normalize_inputs(x, vx, t)
        arf_fx_norm = self.models['arf_x'](x_norm.unsqueeze(1))
        arf_ft_norm = self.models['arf_t'](t_norm.unsqueeze(1))
        arf_fx, arf_ft = self.unified_norm.denormalize_arf_force(arf_fx_norm, arf_ft_norm)
        arf_total = arf_fx.squeeze() * arf_ft.squeeze()
 
        # Stokes factors (separate normalization windows)
        sx_norm, sv_norm, st_norm = self.unified_norm.normalize_inputs_stokes(x, vx, t)
        stokes_fx_norm = self.models['stokes_x'](sx_norm.unsqueeze(1))
        stokes_ft_norm = self.models['stokes_t'](st_norm.unsqueeze(1))
        stokes_fv_norm = self.models['stokes_v'](sv_norm.unsqueeze(1))
        stokes_fx, stokes_ft, stokes_fv = self.unified_norm.denormalize_stokes_force(
            stokes_fx_norm, stokes_ft_norm, stokes_fv_norm
        )
 
        # Stokes drag with acoustic streaming closure
        # F = -3*pi*mu*d*(v - u_air)/C_c
        # u_air ~ -omega*A*cos(kx)*cos(omega*t) encoded by learned factors
        mu = 1.86e-5
        cunningham = 1.0817
        diameter = 2e-6
        omega = 2.0 * np.pi * self.inferred_frequency
        drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
       
        reference_pressure = 20e-6  # Pa
        sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
        rho_0 = 1.225  # air density [kg/m^3]
        c_0 = 340      # sound speed [m/s]
        A = sound_pressure / (rho_0 * c_0 * omega)
       
        u_air = -omega * A * stokes_fx.squeeze() * stokes_ft.squeeze()
       
        stokes_total = -drag_coeff * (vx - u_air)
 
        total_fx = arf_total + stokes_total
        return total_fx  # (N,)
 
def load_individual_models(device='cpu', model_base_path='PINN'):
    """Load five factorized checkpoints + small JSON normalizers."""
    models = {}
    
    # Infer kHz tag from path substring `freq_8k` etc.
    import re
    freq_match = re.search(r'freq_(\d+)k', model_base_path)
    if freq_match:
        inferred_frequency = int(freq_match.group(1)) * 1000
        print(f"  Inferred frequency from path: {inferred_frequency} Hz")
    else:
        inferred_frequency = frequency # Default fallback
        print(f"  Using global default frequency: {inferred_frequency} Hz")
   
    # ARF checkpoints
    try:
        from mechanisms.ARF_PINN_x import ARFNet as ARFNetX, Normalizer as ARFNormalizerX
        from mechanisms.ARF_PINN_t import ARFNetT, Normalizer as ARFNormalizerT
       
        # ARF x
        arf_x_model = ARFNetX(fourier_features=32).to(device)
        arf_x_path = os.path.join(model_base_path, 'arf_model_x.pth')
        if os.path.exists(arf_x_path):
            arf_x_model.load_state_dict(torch.load(arf_x_path, map_location=device))
            models['arf_x'] = arf_x_model
            print(f"  [ok] ARF x weights: {arf_x_path}")
        else:
            print(f"  [missing] ARF x: {arf_x_path}")
       
        # ARF t
        period = 1.0 / inferred_frequency
        arf_t_model = ARFNetT(period_seconds=period).to(device)
        arf_t_path = os.path.join(model_base_path, 'arf_model_t.pth')
        if os.path.exists(arf_t_path):
            arf_t_model.load_state_dict(torch.load(arf_t_path, map_location=device))
            models['arf_t'] = arf_t_model
            print(f"  [ok] ARF t weights: {arf_t_path}")
        else:
            print(f"  [missing] ARF t: {arf_t_path}")
       
        # ARF JSON normalizers
        arf_x_normalizer = ARFNormalizerX()
        arf_x_norm_path = os.path.join(model_base_path, 'arf_model_x_normalization_params.json')
        if os.path.exists(arf_x_norm_path):
            arf_x_normalizer.load(arf_x_norm_path, device)
            models['arf_x_norm'] = arf_x_normalizer
            print(f"  [ok] ARF x normalization JSON")
        else:
            print(f"  [missing] ARF x JSON: {arf_x_norm_path}")
       
        arf_t_normalizer = ARFNormalizerT()
        arf_t_norm_path = os.path.join(model_base_path, 'arf_model_t_normalization_params.json')
        if os.path.exists(arf_t_norm_path):
            arf_t_normalizer.load(arf_t_norm_path, device)
            models['arf_t_norm'] = arf_t_normalizer
            print(f"  [ok] ARF t normalization JSON")
        else:
            print(f"  [missing] ARF t JSON: {arf_t_norm_path}")
           
    except Exception as e:
        print(f"Error while loading ARF checkpoints: {e}")
   
    # Stokes checkpoints
    try:
        from mechanisms.STOKES_PINN_x import StokesNetX, Normalizer as StokesNormalizerX
        from mechanisms.STOKES_PINN_t import StokesNetT, Normalizer as StokesNormalizerT
        from mechanisms.STOKES_PINN_v import StokesNetV, Normalizer as StokesNormalizerV
       
        # Stokes x
        stokes_x_model = StokesNetX(fourier_features=32).to(device)
        stokes_x_path = os.path.join(model_base_path, 'stokes_model_x.pth')
        if os.path.exists(stokes_x_path):
            stokes_x_model.load_state_dict(torch.load(stokes_x_path, map_location=device))
            models['stokes_x'] = stokes_x_model
            print(f"  [ok] Stokes x weights: {stokes_x_path}")
        else:
            print(f"  [missing] Stokes x: {stokes_x_path}")
       
        # Stokes t
        period = 1.0 / inferred_frequency
        stokes_t_model = StokesNetT(period_seconds=period).to(device)
        stokes_t_path = os.path.join(model_base_path, 'stokes_model_t.pth')
        if os.path.exists(stokes_t_path):
            stokes_t_model.load_state_dict(torch.load(stokes_t_path, map_location=device))
            models['stokes_t'] = stokes_t_model
            print(f"  [ok] Stokes t weights: {stokes_t_path}")
        else:
            print(f"  [missing] Stokes t: {stokes_t_path}")
       
        # Stokes v
        stokes_v_model = StokesNetV().to(device)
        stokes_v_path = os.path.join(model_base_path, 'stokes_model_v.pth')
        if os.path.exists(stokes_v_path):
            stokes_v_model.load_state_dict(torch.load(stokes_v_path, map_location=device))
            models['stokes_v'] = stokes_v_model
            print(f"  [ok] Stokes v weights: {stokes_v_path}")
        else:
            print(f"  [missing] Stokes v: {stokes_v_path}")
       
        # Stokes JSON
        stokes_x_normalizer = StokesNormalizerX()
        stokes_x_norm_path = os.path.join(model_base_path, 'stokes_model_x_normalization_params.json')
        if os.path.exists(stokes_x_norm_path):
            with open(stokes_x_norm_path, 'r') as f:
                params = json.load(f)
            stokes_x_normalizer.stats = {
                'x': (torch.tensor(params['x_min'], dtype=torch.float32),
                      torch.tensor(params['x_max'] - params['x_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32),
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_x_norm'] = stokes_x_normalizer
            print(f"  [ok] Stokes x normalization JSON")
        else:
            print(f"  [missing] Stokes x JSON: {stokes_x_norm_path}")
       
        stokes_t_normalizer = StokesNormalizerT()
        stokes_t_norm_path = os.path.join(model_base_path, 'stokes_model_t_normalization_params.json')
        if os.path.exists(stokes_t_norm_path):
            with open(stokes_t_norm_path, 'r') as f:
                params = json.load(f)
            stokes_t_normalizer.stats = {
                't': (torch.tensor(params['t_min'], dtype=torch.float32),
                      torch.tensor(params['t_max'] - params['t_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32),
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_t_norm'] = stokes_t_normalizer
            print(f"  [ok] Stokes t normalization JSON")
        else:
            print(f"  [missing] Stokes t JSON: {stokes_t_norm_path}")
       
        stokes_v_normalizer = StokesNormalizerV()
        stokes_v_norm_path = os.path.join(model_base_path, 'stokes_model_v_normalization_params.json')
        if os.path.exists(stokes_v_norm_path):
            with open(stokes_v_norm_path, 'r') as f:
                params = json.load(f)
            stokes_v_normalizer.stats = {
                'vx': (torch.tensor(params['vx_min'], dtype=torch.float32),
                       torch.tensor(params['vx_max'] - params['vx_min'], dtype=torch.float32)),
                'fx': (torch.tensor(params['force_mu'], dtype=torch.float32),
                       torch.tensor(params['force_sigma'], dtype=torch.float32))
            }
            models['stokes_v_norm'] = stokes_v_normalizer
            print(f"  [ok] Stokes v normalization JSON")
        else:
            print(f"  [missing] Stokes v JSON: {stokes_v_norm_path}")
           
    except Exception as e:
        print(f"Error while loading Stokes checkpoints: {e}")
   
    return models
 
 
def compute_unified_force(positions, velocities, time, models, device='cpu'):
    """Legacy helper: numpy in/out, requires `unified_norm` inside `models` dict."""
    if not all(key in models for key in ['arf_x', 'arf_t', 'stokes_x', 'stokes_t', 'stokes_v']):
        print("Sub-models missing; cannot evaluate forces.")
        return np.zeros_like(positions)
   
    try:
        with torch.no_grad():
            x = torch.tensor(positions[:, 0], dtype=torch.float32, device=device)
            vx = torch.tensor(velocities[:, 0], dtype=torch.float32, device=device)
            t = torch.full_like(x, time, dtype=torch.float32)
           
            x_norm, vx_norm, t_norm = models['unified_norm'].normalize_inputs(x, vx, t)
           
            # ARF factors
            arf_fx_norm = models['arf_x'](x_norm.unsqueeze(1))
            arf_ft_norm = models['arf_t'](t_norm.unsqueeze(1))
            arf_fx, arf_ft = models['unified_norm'].denormalize_arf_force(arf_fx_norm, arf_ft_norm)
            arf_total = arf_fx.squeeze() * arf_ft.squeeze()
           
            # Stokes factors
            stokes_x_norm, stokes_v_norm, stokes_t_norm = models['unified_norm'].normalize_inputs_stokes(x, vx, t)
            stokes_fx_norm = models['stokes_x'](stokes_x_norm.unsqueeze(1))
            stokes_ft_norm = models['stokes_t'](stokes_t_norm.unsqueeze(1))
            stokes_fv_norm = models['stokes_v'](stokes_v_norm.unsqueeze(1))
            stokes_fx, stokes_ft, stokes_fv = models['unified_norm'].denormalize_stokes_force(
                stokes_fx_norm, stokes_ft_norm, stokes_fv_norm)
            mu = 1.86e-5
            cunningham = 1.0817
            diameter = 2e-6
            omega = 2.0 * np.pi * frequency
            drag_coeff = 3.0 * np.pi * mu * diameter / cunningham
           
            reference_pressure = 20e-6  # Pa
            sound_pressure = reference_pressure * (10 ** (sound_pressure_level / 20))
            rho_0 = 1.225
            c_0 = 340
            A = sound_pressure / (rho_0 * c_0 * omega)
           
            u_air = -omega * A * stokes_fx.squeeze() * stokes_ft.squeeze()
           
            stokes_total = -drag_coeff * (vx - u_air)
           
            total_fx = arf_total + stokes_total
            total_fy = torch.zeros_like(total_fx)
           
            return torch.stack([total_fx, total_fy], dim=1).cpu().numpy()
           
    except Exception as e:
        print(f"compute_unified_force failed: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros_like(positions)
 
 
def main():
    """Quick self-test + optional matplotlib animation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
   
    print("Loading factorized checkpoints...")
    model_base_path = 'PINN'
    models = load_individual_models(device, model_base_path)
   
    # Aggregate JSON stats
    unified_norm = UnifiedNormalizer()
    model_paths = {
        'arf_x_norm': os.path.join(model_base_path, 'arf_model_x_normalization_params.json'),
        'arf_t_norm': os.path.join(model_base_path, 'arf_model_t_normalization_params.json'),
        'stokes_x_norm': os.path.join(model_base_path, 'stokes_model_x_normalization_params.json'),
        'stokes_t_norm': os.path.join(model_base_path, 'stokes_model_t_normalization_params.json'),
        'stokes_v_norm': os.path.join(model_base_path, 'stokes_model_v_normalization_params.json')
    }
    unified_norm.load_from_individual_models(model_paths, device)
    models['unified_norm'] = unified_norm
 
    phys_model = PhysicalUnifiedModel(models, unified_norm, device=device)
   
    required_models = ['arf_x', 'arf_t', 'stokes_x', 'stokes_t', 'stokes_v']
    missing_models = [model for model in required_models if model not in models]
    if missing_models:
        print(f"Missing checkpoints: {missing_models}")
        return
   
    print("All required weights present.")
   
    print("\nScalar test...")
   
    N = 100
    x_test = torch.linspace(0.001, 0.05, N, device=device)  # 1mm to 50mm
    vx_test = torch.linspace(-0.05, 0.05, N, device=device)  # -5cm/s to 5cm/s
    t_test = torch.linspace(0, 1.0/frequency, N, device=device)
   
    positions = torch.stack([x_test, torch.zeros_like(x_test)], dim=1).cpu().numpy()
    velocities = torch.stack([vx_test, torch.zeros_like(vx_test)], dim=1).cpu().numpy()
    time = 0.0
   
    forces = compute_unified_force(positions, velocities, time, models, device)
   
    print(f"Evaluated {N} samples")
    print(f"Fx range: [{forces[:, 0].min():.2e}, {forces[:, 0].max():.2e}] N")
    print(f"Fx mean: {forces[:, 0].mean():.2e} N")
   
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        print("\nBuilding demo animation (0–0.1 ms, 10 µs steps, looping)...")
 
        x_line = torch.linspace(0.0, c_0 / frequency, 200, device=device)
        vx_line = torch.linspace(-0.1, 0.1, 200, device=device)
        pos_x = torch.stack([x_line, torch.zeros_like(x_line)], dim=1).cpu().numpy()
        pos_zero = np.zeros((vx_line.shape[0], 2), dtype=pos_x.dtype)
        vel_zero = np.zeros((pos_x.shape[0], 2), dtype=pos_x.dtype)
        vel_v = torch.stack([vx_line, torch.zeros_like(vx_line)], dim=1).cpu().numpy()
        # Scatter particles for illustration
        rng = np.random.default_rng(42)
        Np = 300
        part_pos = np.zeros((Np, 2), dtype=pos_x.dtype)
        part_pos[:, 0] = rng.uniform(0.0, float(c_0 / frequency), size=Np)
        part_vel = np.zeros_like(part_pos)
 
        t_frames = np.arange(0.0, 1e-4 + 1e-12, 1e-5)
       
        t_plot = np.linspace(0.0, 1e-4, 200)
 
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        ax_x, ax_v, ax_t, ax_p = axes.flatten()
 
        line_x, = ax_x.plot((x_line * 1000).cpu().numpy(), np.zeros_like(x_line.cpu().numpy()), 'b-')
        ax_x.set_xlabel('x (mm)'); ax_x.set_ylabel('Fx (N)'); ax_x.set_title('Fx vs x (vx=0)'); ax_x.grid(True, alpha=0.3)
 
        line_v, = ax_v.plot(vx_line.cpu().numpy(), np.zeros_like(vx_line.cpu().numpy()), 'g-')
        ax_v.set_xlabel('vx (m/s)'); ax_v.set_ylabel('Fx (N)'); ax_v.set_title('Fx vs vx (x=0)'); ax_v.grid(True, alpha=0.3)
 
        t_us_plot = t_plot * 1e6
        line_t, = ax_t.plot(t_us_plot, np.zeros_like(t_us_plot), 'r-')
        ax_t.set_xlabel('t (µs)'); ax_t.set_ylabel('Fx (N, mean)'); ax_t.set_title('Fx vs t (x=0, vx=0)'); ax_t.grid(True, alpha=0.3)
 
        scat_p = ax_p.scatter(part_pos[:, 0] * 1000.0, np.zeros_like(part_pos[:, 0]), s=1, c='purple', alpha=0.7)
        ax_p.set_xlabel('x (mm)'); ax_p.set_ylabel('Fx (N)'); ax_p.set_title('Particle Forces'); ax_p.grid(True, alpha=0.3)
 
        def update(frame_idx):
            tt = float(t_frames[frame_idx])
            # Fx(x)
            x_tensor = torch.tensor(pos_x[:, 0], device=device, dtype=torch.float32)
            vx_zero = torch.zeros_like(x_tensor)
            t_tensor = torch.full_like(x_tensor, tt)
            Fx_x = phys_model(x_tensor, vx_zero, t_tensor).cpu().numpy()
            line_x.set_ydata(Fx_x)
            # Fx(vx)
            vx_tensor = torch.tensor(vel_v[:, 0], device=device, dtype=torch.float32)
            x_zero = torch.zeros_like(vx_tensor)
            t_v = torch.full_like(vx_tensor, tt)
            Fx_v = phys_model(x_zero, vx_tensor, t_v).cpu().numpy()
            line_v.set_ydata(Fx_v)
            # Mean Fx(t) via dense t_plot samples
            Fx_t_vals = []
            for ttmp in t_plot:
                t_tmp_tensor = torch.full_like(x_zero, float(ttmp))
                Fx_tmp = phys_model(x_zero, vx_zero, t_tmp_tensor).cpu().numpy()
                Fx_t_vals.append(Fx_tmp.mean())
            line_t.set_ydata(np.array(Fx_t_vals))
            xp = torch.tensor(part_pos[:, 0], device=device, dtype=torch.float32)
            vp = torch.tensor(part_vel[:, 0], device=device, dtype=torch.float32)
            tp = torch.full_like(xp, tt)
            Fx_p = phys_model(xp, vp, tp).cpu().numpy()
            scat_p.set_offsets(np.c_[part_pos[:, 0] * 1000.0, Fx_p])
            ax_x.relim(); ax_x.autoscale_view()
            ax_v.relim(); ax_v.autoscale_view()
            ax_t.relim(); ax_t.autoscale_view()
            y_min = np.min(Fx_p)
            y_max = np.max(Fx_p)
            if y_min == y_max:
                y_pad = 1e-12 if y_min == 0 else abs(y_min) * 0.05
                y_min -= y_pad
                y_max += y_pad
            else:
                pad = (y_max - y_min) * 0.05
                y_min -= pad
                y_max += pad
            ax_p.set_ylim(y_min, y_max)
            return line_x, line_v, line_t, scat_p
 
        anim = FuncAnimation(fig, update, frames=len(t_frames), interval=200, blit=False, repeat=True)
        plt.show()
        print("Animation ready (looping).")
    except Exception as _e:
        print(f"Animation skipped: {_e}")
   
    unified_model = UnifiedPINN(fourier_features=32).to(device)
    torch.save(unified_model.state_dict(), 'PINN/unified_model.pth')
   
    unified_norm_params = {
        'x_min': unified_norm.stats['x_min'],
        'x_max': unified_norm.stats['x_max'],
        'vx_min': unified_norm.stats['vx_min'],
        'vx_max': unified_norm.stats['vx_max'],
        't_min': unified_norm.stats['t_min'],
        't_max': unified_norm.stats['t_max'],
        'arf_fx_mu': unified_norm.stats['arf_fx_mu'],
        'arf_fx_sigma': unified_norm.stats['arf_fx_sigma'],
        'arf_ft_mu': unified_norm.stats['arf_ft_mu'],
        'arf_ft_sigma': unified_norm.stats['arf_ft_sigma'],
        'stokes_fx_mu': unified_norm.stats['stokes_fx_mu'],
        'stokes_fx_sigma': unified_norm.stats['stokes_fx_sigma'],
        'stokes_ft_mu': unified_norm.stats['stokes_ft_mu'],
        'stokes_ft_sigma': unified_norm.stats['stokes_ft_sigma'],
        'stokes_fv_mu': unified_norm.stats['stokes_fv_mu'],
        'stokes_fv_sigma': unified_norm.stats['stokes_fv_sigma']
    }
   
    with open('PINN/unified_model_normalization_params.json', 'w') as f:
        json.dump(unified_norm_params, f, indent=2)
   
    print("\nWrote PINN/unified_model.pth and PINN/unified_model_normalization_params.json")
 
 
if __name__ == "__main__":
    main()