# particle_initialization.py
import numpy as np
import math


def _min_distance_random_sampling(width, height, min_dist, target_count, rng):
    """
    Uniform random samples with pairwise separation >= min_dist (brute-force check).

    :param width: domain width (x extent) [m]
    :param height: domain height (y extent) [m]
    :param min_dist: minimum center-to-center distance [m]
    :param target_count: number of points
    :param rng: NumPy Generator
    :return: array of shape (target_count, 2)
    """
    if target_count == 0:
        return np.zeros((0, 2), dtype=float)

    print(f"Generating {target_count} particles; minimum separation {min_dist*1e6:.2f} µm")
    print(f"Domain: {width*1000:.2f} mm x {height*1000:.2f} mm")
    
    samples = []
    min_dist_sq = min_dist * min_dist
    
    max_capacity = int((width * height) / (np.pi * (min_dist / 2) ** 2))
    print(f"Theoretical capacity (order of magnitude): {max_capacity} particles")
    if target_count > max_capacity * 0.9:
        print(f"Warning: N={target_count} is >90% of the crude packing bound; placement may fail.")

    max_total_attempts = max(1000 * target_count, 100000)
    attempts = 0
    consecutive_failures = 0

    while len(samples) < target_count:
        if attempts >= max_total_attempts:
            raise RuntimeError(
                f"Exceeded {max_total_attempts} attempts while placing {target_count} particles. "
                f"Placed {len(samples)}. Reduce N or enlarge the domain."
            )
        
        candidate_x = float(rng.uniform(0.0, width))
        candidate_y = float(rng.uniform(0.0, height))
        candidate = np.array([candidate_x, candidate_y], dtype=np.float64)
        attempts += 1

        too_close = False
        if len(samples) > 0:
            existing = np.array(samples, dtype=np.float64)
            dx = existing[:, 0] - candidate_x
            dy = existing[:, 1] - candidate_y
            dist_sq = dx * dx + dy * dy
            min_existing_dist_sq = np.min(dist_sq)
            
            if min_existing_dist_sq < min_dist_sq:
                too_close = True
                consecutive_failures += 1
            else:
                consecutive_failures = 0

        if too_close:
            continue

        samples.append([candidate_x, candidate_y])
        
        if len(samples) % max(1, target_count // 20) == 0 or len(samples) == target_count:
            print(f"Placed {len(samples)}/{target_count} (attempts: {attempts}, accept rate: {len(samples)/attempts*100:.2f}%)")

    samples_array = np.array(samples, dtype=np.float64)
    
    print(f"Final check: pairwise distances for {len(samples_array)} particles...")
    try:
        from scipy.spatial.distance import pdist
        distances = pdist(samples_array)
        min_actual_dist = float(np.min(distances))
        violations = distances < (min_dist - 1e-12)
        violation_count = int(np.sum(violations))
        
        if violation_count > 0:
            from scipy.spatial.distance import squareform
            dist_matrix = squareform(distances)
            viol_pairs = []
            for i in range(len(samples_array)):
                for j in range(i + 1, len(samples_array)):
                    if dist_matrix[i, j] < min_dist - 1e-12:
                        viol_pairs.append((i, j, dist_matrix[i, j]))
                        if len(viol_pairs) >= 10:
                            break
                if len(viol_pairs) >= 10:
                    break
            
            print(f"Found {violation_count} violating pairs!")
            for i, (idx1, idx2, dist) in enumerate(viol_pairs[:10]):
                print(f"  #{i+1}: particles {idx1} ({samples_array[idx1]*1000}) and {idx2} ({samples_array[idx2]*1000}) "
                      f"distance {dist*1e6:.4f} µm < {min_dist*1e6:.4f} µm")
            
            raise RuntimeError(
                f"Validation failed: {violation_count} violations; min distance {min_actual_dist*1e6:.4f} µm "
                f"(required {min_dist*1e6:.4f} µm)"
            )
        
        print(f"OK: all {len(samples_array)} separations >= {min_dist*1e6:.4f} µm")
        print(f"  Minimum: {min_actual_dist*1e6:.4f} µm; mean pairwise: {np.mean(distances)*1e6:.4f} µm")
        
    except ImportError:
        print("Warning: scipy not installed; using O(N^2) distance check")
        min_actual_dist = float('inf')
        violation_count = 0
        
        for i in range(len(samples_array)):
            for j in range(i + 1, len(samples_array)):
                dx = samples_array[i, 0] - samples_array[j, 0]
                dy = samples_array[i, 1] - samples_array[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_actual_dist = min(min_actual_dist, dist)
                if dist < min_dist - 1e-12:
                    violation_count += 1
                    if violation_count <= 10:
                        print(f"Violation: pair ({i},{j}) distance {dist*1e6:.4f} µm < {min_dist*1e6:.4f} µm")
        
        if violation_count > 0:
            raise RuntimeError(
                f"Validation failed: {violation_count} violations; min distance {min_actual_dist*1e6:.4f} µm "
                f"(required {min_dist*1e6:.4f} µm)"
            )
        
        print(f"OK: all separations >= {min_dist*1e6:.4f} µm (min observed {min_actual_dist*1e6:.4f} µm)")

    return samples_array


def initialize_particles(N=0, domain_size=(0.01, 0.01), diameter=2e-6, density=2000, init_mode='linear'):
    """
    Initialize an ensemble of particles.

    :param N: count
    :param domain_size: (Lx, Ly) [m]
    :param diameter: nominal diameter [m] (ignored if radii set in mode)
    :param density: material density [kg/m^3]
    :param init_mode:
        - 'linear': uniform spacing along x at mid-height
        - 'linear_poly': linear x, diameters uniform in [0.1, 10] µm
        - 'random': Poisson-disk-like rejection sampling (≥2 µm separation)
        - 'uniform': near-uniform grid + fill
        - 'sample': two vertical stacks at x = 8.5 mm and 25.5 mm
        - 'normal': truncated-Gaussian quantiles along x, y = 0.01 m
    :return: positions (N,2), velocities (N,2), radii (N,), mass (N,)
    """
    Lx, Ly = domain_size
    
    radii = None

    if init_mode == 'linear':
        x_positions = np.linspace(0, Lx, N, endpoint=True)
        y_position = Ly / 2
        positions = np.column_stack((x_positions, np.full(N, y_position)))
    elif init_mode == 'linear_poly':
        x_positions = np.linspace(0, Lx, N, endpoint=True)
        y_position = Ly / 2
        positions = np.column_stack((x_positions, np.full(N, y_position)))
        rng = np.random.default_rng()
        diameters_poly = rng.uniform(0.1e-6, 10e-6, size=N)
        radii = diameters_poly / 2.0
        
    elif init_mode == 'random':
        min_center_distance = 2e-6
        rng = np.random.default_rng()
        positions = _min_distance_random_sampling(Lx, Ly, min_center_distance, N, rng)
        
    elif init_mode == 'uniform':
        grid_size = int(np.ceil(np.sqrt(N)))
        x_grid = np.linspace(0, Lx, grid_size, endpoint=False)
        y_grid = np.linspace(0, Ly, grid_size, endpoint=False)
        
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        
        if len(grid_points) > N:
            indices = np.random.choice(len(grid_points), N, replace=False)
            positions = grid_points[indices]
        else:
            positions = grid_points.copy()
            additional_N = N - len(positions)
            additional_x = np.random.uniform(0, Lx, additional_N)
            additional_y = np.random.uniform(0, Ly, additional_N)
            additional_positions = np.column_stack((additional_x, additional_y))
            positions = np.vstack((positions, additional_positions))
    elif init_mode == 'sample':
        x_left = 0.0085
        x_right = 0.0255
        n_left = N // 2
        n_right = N - n_left

        if n_left > 0:
            y_left = np.linspace(0, Ly, n_left, endpoint=False)
            left_positions = np.column_stack((np.full(n_left, x_left), y_left))
        else:
            left_positions = np.empty((0, 2))

        if n_right > 0:
            y_right = np.linspace(0, Ly, n_right, endpoint=False)
            right_positions = np.column_stack((np.full(n_right, x_right), y_right))
        else:
            right_positions = np.empty((0, 2))

        positions = np.vstack((left_positions, right_positions)) if N > 0 else np.zeros((0, 2))
    elif init_mode == 'normal':
        mu_x = Lx / 2.0
        sigma_x = Lx / 10.0

        def _phi(x):
            x = np.asarray(x, dtype=float)
            erf_vec = np.vectorize(math.erf)
            return 0.5 * (1.0 + erf_vec(x / np.sqrt(2.0)))

        def _phi_inv(p):
            # Acklam's approximation to Phi^{-1}, valid for p in (0,1)
            a = np.array([-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
                          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00])
            b = np.array([-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
                          6.680131188771972e+01, -1.328068155288572e+01])
            c = np.array([-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00])
            d = np.array([ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                           3.754408661907416e+00])

            plow = 0.02425
            phigh = 1 - plow
            p = np.asarray(p)
            x = np.empty_like(p, dtype=float)

            mask_low = p < plow
            if np.any(mask_low):
                q = np.sqrt(-2.0 * np.log(p[mask_low]))
                x[mask_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

            mask_mid = (p >= plow) & (p <= phigh)
            if np.any(mask_mid):
                q = p[mask_mid] - 0.5
                r = q*q
                x[mask_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

            mask_high = p > phigh
            if np.any(mask_high):
                q = np.sqrt(-2.0 * np.log(1.0 - p[mask_high]))
                x[mask_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

            return x

        a = (0.0 - mu_x) / sigma_x
        b = (Lx - mu_x) / sigma_x
        Fa = _phi(a)
        Fb = _phi(b)

        p = (np.arange(N, dtype=float) + 0.5) / max(N, 1)
        u = Fa + p * (Fb - Fa)
        z = _phi_inv(u)
        x_positions = mu_x + sigma_x * z
        eps = np.finfo(float).eps
        x_positions = np.clip(x_positions, 0.0, max(0.0, Lx - eps))

        y_positions = np.ones(N) * 0.01
        positions = np.column_stack((x_positions, y_positions))
    else:
        raise ValueError(f"Unknown init_mode={init_mode!r}. Use 'linear', 'random', 'uniform', 'sample', 'normal'.")

    velocities = np.zeros_like(positions)
    if radii is None:
        radii = np.ones(N) * (diameter / 2)
    mass = density * 4/3 * np.pi * radii**3
    
    return positions, velocities, radii, mass


def test_initialization_modes():
    """Smoke-plot a few layout modes."""
    import matplotlib.pyplot as plt
    
    N = 100
    domain_size = (0.01, 0.01)
    diameter = 2e-6
    density = 2000
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['linear', 'random', 'uniform']
    mode_names = ['Linear', 'Random (min dist)', 'Grid / uniform']
    
    for i, (mode, name) in enumerate(zip(modes, mode_names)):
        positions, velocities, radii, mass = initialize_particles(
            N=N, domain_size=domain_size, diameter=diameter, 
            density=density, init_mode=mode
        )
        
        axes[i].scatter(positions[:, 0] * 1000, positions[:, 1] * 1000, 
                       s=20, alpha=0.7, c='blue')
        axes[i].set_xlim(0, domain_size[0] * 1000)
        axes[i].set_ylim(0, domain_size[1] * 1000)
        axes[i].set_xlabel('x (mm)')
        axes[i].set_ylabel('y (mm)')
        axes[i].set_title(f'{name} (N={N})')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    print("Initialization mode smoke test:")
    print(f"Domain: {domain_size[0]*1000:.1f} mm x {domain_size[1]*1000:.1f} mm")
    print(f"N: {N}")
    print(f"Diameter: {diameter*1e6:.1f} µm")
    print(f"Density: {density} kg/m^3")


if __name__ == "__main__":
    test_initialization_modes()
