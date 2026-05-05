import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List


def detect_collisions(positions: np.ndarray, radii: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find overlapping particle pairs (contact or overlap).

    Args:
        positions: (N, 2)
        radii: (N,)

    Returns:
        List of index pairs (i, j) with i < j.
    """
    collisions = []
    N = len(positions)

    tree = cKDTree(positions)

    for i in range(N):
        collision_radius = radii[i] + radii.max()
        neighbors = tree.query_ball_point(positions[i], collision_radius)

        for j in neighbors:
            if i < j:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= (radii[i] + radii[j]):
                    collisions.append((i, j))

    return collisions


def is_head_on_collision(
    pos1: np.ndarray,
    pos2: np.ndarray,
    vel1: np.ndarray,
    vel2: np.ndarray,
    radius1: float,
    radius2: float,
) -> bool:
    """
    Approximate head-on / imminent collision along the line of centres.
    """
    rel_pos = pos2 - pos1
    distance = np.linalg.norm(rel_pos)

    if distance <= (radius1 + radius2):
        return True

    rel_vel = vel2 - vel1
    dot_product = np.dot(rel_pos, rel_vel)

    if dot_product < 0:
        rel_vel_mag = np.linalg.norm(rel_vel)
        if rel_vel_mag > 1e-10:
            collision_time = (distance - radius1 - radius2) / rel_vel_mag
            if collision_time <= 0.001:
                return True

    return False


def calculate_collision_velocity(
    vel1: np.ndarray,
    vel2: np.ndarray,
    mass1: float,
    mass2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perfectly inelastic collision: equal post-collision velocity (momentum-conserving).
    """
    total_mass = mass1 + mass2
    momentum = mass1 * vel1 + mass2 * vel2
    common_velocity = momentum / total_mass
    return common_velocity, common_velocity


def merge_particles(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    mass: np.ndarray,
    collision_pairs: List[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge pairs in collision_pairs (assumes velocity already updated)."""
    if not collision_pairs:
        return positions, velocities, radii, mass

    collision_pairs.sort(key=lambda pair: mass[pair[0]] + mass[pair[1]], reverse=True)

    particles_to_remove = set()

    for i, j in collision_pairs:
        if i in particles_to_remove or j in particles_to_remove:
            continue

        total_mass = mass[i] + mass[j]
        merged_position = (mass[i] * positions[i] + mass[j] * positions[j]) / total_mass
        merged_velocity = (mass[i] * velocities[i] + mass[j] * velocities[j]) / total_mass
        merged_radius = np.cbrt(radii[i] ** 3 + radii[j] ** 3)

        positions[i] = merged_position
        velocities[i] = merged_velocity
        radii[i] = merged_radius
        mass[i] = total_mass
        particles_to_remove.add(j)

    if particles_to_remove:
        keep_indices = [i for i in range(len(positions)) if i not in particles_to_remove]
        positions = positions[keep_indices]
        velocities = velocities[keep_indices]
        radii = radii[keep_indices]
        mass = mass[keep_indices]

    return positions, velocities, radii, mass


def apply_collision_physics(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    mass: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect collisions, apply inelastic velocity, then merge clusters."""
    if len(positions) < 2:
        return positions, velocities, radii, mass

    collision_pairs = detect_collisions(positions, radii)

    head_on_collisions = []
    for i, j in collision_pairs:
        if is_head_on_collision(
            positions[i],
            positions[j],
            velocities[i],
            velocities[j],
            radii[i],
            radii[j],
        ):
            head_on_collisions.append((i, j))

    if head_on_collisions:
        for i, j in head_on_collisions:
            vel1_new, vel2_new = calculate_collision_velocity(
                velocities[i], velocities[j], mass[i], mass[j]
            )
            velocities[i] = vel1_new
            velocities[j] = vel2_new

        positions, velocities, radii, mass = merge_particles(
            positions, velocities, radii, mass, head_on_collisions
        )

    return positions, velocities, radii, mass


def calculate_collision_energy(
    vel1: np.ndarray,
    vel2: np.ndarray,
    mass1: float,
    mass2: float,
) -> float:
    """Total kinetic energy before impact."""
    ke1 = 0.5 * mass1 * np.linalg.norm(vel1) ** 2
    ke2 = 0.5 * mass2 * np.linalg.norm(vel2) ** 2
    return ke1 + ke2


def calculate_merged_particle_properties(
    positions: np.ndarray,
    velocities: np.ndarray,
    radii: np.ndarray,
    mass: np.ndarray,
    collision_pairs: List[Tuple[int, int]],
) -> dict:
    """Diagnostics for merge statistics."""
    if not collision_pairs:
        return {}

    stats = {
        "total_collisions": len(collision_pairs),
        "collision_pairs": collision_pairs,
        "energy_loss": 0.0,
        "merged_particles": [],
    }

    for i, j in collision_pairs:
        initial_ke = calculate_collision_energy(velocities[i], velocities[j], mass[i], mass[j])
        merged_vel = (mass[i] * velocities[i] + mass[j] * velocities[j]) / (mass[i] + mass[j])
        merged_mass = mass[i] + mass[j]
        final_ke = 0.5 * merged_mass * np.linalg.norm(merged_vel) ** 2
        energy_loss = initial_ke - final_ke
        stats["energy_loss"] += energy_loss

        stats["merged_particles"].append(
            {
                "particle1": {"index": i, "mass": mass[i], "radius": radii[i]},
                "particle2": {"index": j, "mass": mass[j], "radius": radii[j]},
                "merged_mass": merged_mass,
                "merged_radius": np.cbrt(radii[i] ** 3 + radii[j] ** 3),
                "energy_loss": energy_loss,
            }
        )

    return stats
