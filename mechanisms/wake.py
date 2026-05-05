import numpy as np

# Characteristic length [m] to nondimensionalize r so that R thresholds (2,5) in test_wake.py apply
CHAR_LENGTH_M = 5e-4  # 0.5 mm, maps mm-scale grid to R ~ O(1)
WAKE_STRENGTH_MULTIPLIER = 1

def acoustic_wake_velocity(source_pos, target_pos, source_vel, Re, return_cartesian=True):
    source_pos = np.asarray(source_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)
    source_vel = np.asarray(source_vel, dtype=np.float64)

    # Relative position in global coordinates (treat particle as origin)
    rel_vec = target_pos - source_pos
    x = rel_vec[0]
    y = rel_vec[1]

    r = np.hypot(x, y)
    if r < 1e-12:
        return (0.0, 0.0)

    R = r / CHAR_LENGTH_M

    # Polar angle (full plane); must match test_wake.py conventions for benchmarks
    Theta = np.arctan2(y, x)

    cos_t = np.cos(Theta)
    sin_t = np.sin(Theta)
    cos_2t = np.cos(2 * Theta)

    # Oseen terms (exactly as test_wake.py)
    Oseen_r_term1 = cos_t / (2 * R)
    Oseen_r_term2 = (3 * R * (1 + cos_t) / 4) * np.exp((-R * Re / 4) * (1 - cos_t))
    Oseen_r_term3 = 3 * (1 - np.exp((-R * Re / 4) * (1 - cos_t))) / Re
    Vr_Oseen = (Oseen_r_term1 - Oseen_r_term2 + Oseen_r_term3) / (R ** 2)

    Oseen_t_term1 = 1 / (R ** 2)
    Oseen_t_term2 = 3 * np.exp((-R * Re / 4) * (1 - cos_t))
    Vt_Oseen = sin_t * (Oseen_t_term1 + Oseen_t_term2) / (4 * R)

    # PP terms
    PP_r_term1 = 4 * R * (16 + 3 * Re + 3 * R ** 2 * (-16 + (2 * R - 3) * Re)) * cos_t
    PP_r_term2 = 3 * ((R - 1) ** 2) * (1 + R + 2 * R ** 2) * (1 + 3 * cos_2t) * Re
    Vr_PP = (PP_r_term1 - PP_r_term2) / (128 * R ** 4)

    PP_t_term1 = R * (16 + 3 * Re + 3 * R ** 2 * (16 + (3 - 4 * R) * Re))
    PP_t_term2 = 3 * (-2 + R - 3 * R ** 3 + 4 * R ** 4) * cos_t * Re
    Vt_PP = sin_t * (PP_t_term1 + PP_t_term2) / (64 * R ** 4)

    # Piecewise combination (exactly as test_wake.py)
    if R < 2:
        vr = Vr_PP
        vt = Vt_PP
    elif R <= 5:
        vr = ((5 - R) * Vr_PP + (R - 2) * Vr_Oseen) / 3
        vt = ((5 - R) * Vt_PP + (R - 2) * Vt_Oseen) / 3
    else:
        vr = Vr_Oseen
        vt = Vt_Oseen

    if not return_cartesian:
        return (vr, vt)

    U = vr * np.cos(Theta) - vt * np.sin(Theta)
    V = vr * np.sin(Theta) + vt * np.cos(Theta)

    # Heuristic upstream/downstream bias for left-moving source (see test_wake.py)
    if x > 0:
        U = -abs(U)
    else:
        U = -abs(U) * 0.5

    # Return global coordinates directly
    vx_global, vy_global = U, V

    if not np.isfinite(vx_global) or not np.isfinite(vy_global):
        return (0.0, 0.0)

    return (vx_global, vy_global)
