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
    """根据声场计算压力（或压力平方）的空间梯度，并施加声涌辐射力。
    方向严格为负梯度方向：F = -k * ∇(P) 或 F = -k * ∇(P^2)（驻波）。"""
    X, Y, P = compute_sound_field(domain_size=domain_size, resolution=resolution, time=time)
    Nx, Ny = resolution
    Lx, Ly = domain_size
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    # 驻波用 -∇(p^2)，行波用 -∇p
    if is_standing_wave:
        P_field = P**2
    else:
        P_field = P

    dP_dy, dP_dx = np.gradient(P_field, dy, dx, edge_order=2)

    grad = np.zeros_like(positions)
    for i, (x, y) in enumerate(positions):
        # 注意：np.gradient 返回 d/dy 在第一项，d/dx 在第二项
        gx = bilinear_interpolate(dP_dx, x, y, dx, dy)
        gy = bilinear_interpolate(dP_dy, x, y, dx, dy)
        grad[i, 0] = gx
        grad[i, 1] = gy

    # 力方向严格沿负梯度
    arf_force = -arf_strength * grad

    # 加速度 = F / m
    accelerations = arf_force / mass[:, None]

    # 显式欧拉更新
    velocities = velocities + accelerations * dt
    positions = positions + velocities * dt

    return positions, velocities
