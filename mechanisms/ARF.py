import numpy as np

def bilinear_interpolate(grid, x, y, dx, dy):
    Nx = grid.shape[1]
    Ny = grid.shape[0]
    ix = np.clip(x / dx, 0, Nx - 2)
    iy = np.clip(y / dy, 0, Ny - 2)
    x0 = np.floor(ix).astype(int)
    y0 = np.floor(iy).astype(int)
    x1 = np.clip(x0 + 1, 0, Nx - 1)
    y1 = np.clip(y0 + 1, 0, Ny - 1)
    wx = ix - x0
    wy = iy - y0
    # grid[y, x]，所以先行后列
    return (
        (1-wx)*(1-wy)*grid[y0, x0] +
        wx*(1-wy)*grid[y0, x1] +
        (1-wx)*wy*grid[y1, x0] +
        wx*wy*grid[y1, x1]
    )

def compute_pressure_gradient_and_apply_arf(
    positions, velocities, mass, dt, domain_size, resolution, time, compute_sound_field, arf_strength=1e-23, is_standing_wave=False
):
    """使用瞬时声压场的梯度计算声涌辐射力：F = -k * ∇P。
    这会使粒子始终沿着压力下降方向运动（趋向低压区）。"""
    X, Y, P = compute_sound_field(domain_size=domain_size, resolution=resolution, time=time)
    Nx, Ny = resolution
    Lx, Ly = domain_size
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # 使用瞬时压力场
    dP_dy, dP_dx = np.gradient(P, dy, dx, edge_order=2)

    grad = np.zeros_like(positions)
    for i, (x, y) in enumerate(positions):
        gx = bilinear_interpolate(dP_dx, x, y, dx, dy)
        gy = bilinear_interpolate(dP_dy, x, y, dx, dy)
        grad[i, 0] = gx
        grad[i, 1] = gy

    # 力方向严格沿负梯度（指向更低的瞬时压力）
    arf_force = -arf_strength * grad

    # 加速度 = F / m
    accelerations = arf_force / mass[:, None]

    # 显式欧拉更新
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt

    return positions, velocities
