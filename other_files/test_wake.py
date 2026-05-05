import numpy as np
import matplotlib.pyplot as plt


def compute_disturbance_velocity_field_np(Re=1.0, r_range=(0.5, 12), theta_range=(-np.pi, np.pi),
                                          r_points=150, theta_points=150):

    r_vals = np.linspace(r_range[0], r_range[1], r_points)
    theta_vals = np.linspace(theta_range[0], theta_range[1], theta_points)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    cos_2t = np.cos(2 * Theta)

    # Oseen branch
    Oseen_r_term1 = cos_t / (2 * R)
    Oseen_r_term2 = (3 * R * (1 + cos_t) / 4) * np.exp((-R * Re / 4) * (1 - cos_t))
    Oseen_r_term3 = 3 * (1 - np.exp((-R * Re / 4) * (1 - cos_t))) / Re
    Vr_Oseen = (Oseen_r_term1 - Oseen_r_term2 + Oseen_r_term3) / (R ** 2)

    Oseen_t_term1 = 1 / (R ** 2)
    Oseen_t_term2 = 3 * np.exp((-R * Re / 4) * (1 - cos_t))
    Vt_Oseen = sin_t * (Oseen_t_term1 + Oseen_t_term2) / (4 * R)

    # Proudman–Pearson branch
    PP_r_term1 = 4 * R * (16 + 3 * Re + 3 * R ** 2 * (-16 + (2 * R - 3) * Re)) * cos_t
    PP_r_term2 = 3 * ((R - 1) ** 2) * (1 + R + 2 * R ** 2) * (1 + 3 * cos_2t) * Re
    Vr_PP = (PP_r_term1 - PP_r_term2) / (128 * R ** 4)

    PP_t_term1 = R * (16 + 3 * Re + 3 * R ** 2 * (16 + (3 - 4 * R) * Re))
    PP_t_term2 = 3 * (-2 + R - 3 * R ** 3 + 4 * R ** 4) * cos_t * Re
    Vt_PP = sin_t * (PP_t_term1 + PP_t_term2) / (64 * R ** 4)

    # Blended segments (same breakpoints as DEM_wake-style mixing)
    Vr = np.zeros_like(R)
    Vt = np.zeros_like(R)

    mask1 = R < 2
    Vr[mask1] = Vr_PP[mask1]
    Vt[mask1] = Vt_PP[mask1]

    mask2 = (R >= 2) & (R <= 5)
    Vr[mask2] = ((5 - R[mask2]) * Vr_PP[mask2] + (R[mask2] - 2) * Vr_Oseen[mask2]) / 3
    Vt[mask2] = ((5 - R[mask2]) * Vt_PP[mask2] + (R[mask2] - 2) * Vt_Oseen[mask2]) / 3

    mask3 = R > 5
    Vr[mask3] = Vr_Oseen[mask3]
    Vt[mask3] = Vt_Oseen[mask3]

    # Polar -> Cartesian velocity components
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    U = Vr * np.cos(Theta) - Vt * np.sin(Theta)
    V = Vr * np.sin(Theta) + Vt * np.cos(Theta)

    return X, Y, U, V


X, Y, U, V = compute_disturbance_velocity_field_np(Re=1.0)

step = 6  # subsample arrows

X_sparse = X[::step, ::step]
Y_sparse = Y[::step, ::step]
U_sparse = U[::step, ::step]
V_sparse = V[::step, ::step]

radius = 0.5
distance = np.sqrt(X_sparse**2 + Y_sparse**2)
mask = distance > 2 * radius

X_masked = X_sparse[mask]
Y_masked = Y_sparse[mask]
U_masked = U_sparse[mask]
V_masked = V_sparse[mask]

plt.figure(figsize=(12, 8))
plt.quiver(X_masked, Y_masked, U_masked, V_masked,
           color='blue', scale=3, scale_units='width', angles='xy', width=0.002)

plt.gca().add_patch(plt.Circle((0, 0), radius, color='black', fill=False))

plt.xlim(-12, 12)
plt.ylim(-12, 12)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Disturbance Velocity Field (Extended Range)')
plt.grid(True)
plt.show()
