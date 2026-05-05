import numpy as np

def check_collisions(positions, radii):
    """
    Enumerate overlapping disk pairs (center distance < r_i + r_j).

    Returns:
        list of (i, j) with i < j
    """
    N = len(positions)
    collisions = []
    
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < (radii[i] + radii[j]):
                collisions.append((i, j))
    
    return collisions

def merge_particles(positions, velocities, radii, mass, collision_pairs):
    """
    Perfectly inelastic merge for each pair (updates particle i, marks j removed).

    Returns:
        Updated arrays and a set of indices to delete.
    """
    N = len(positions)
    to_delete = set()
    
    for i, j in collision_pairs:
        if i in to_delete or j in to_delete:
            continue
            
        total_mass = mass[i] + mass[j]
        center_of_mass = (positions[i] * mass[i] + positions[j] * mass[j]) / total_mass
        total_momentum = velocities[i] * mass[i] + velocities[j] * mass[j]
        merged_velocity = total_momentum / total_mass
        
        # 2-D area-conserving heuristic with visualization scaling
        merged_radius = np.sqrt(radii[i]**2 + radii[j]**2) * 1.2
        
        positions[i] = center_of_mass
        velocities[i] = merged_velocity
        radii[i] = merged_radius
        mass[i] = total_mass
        
        to_delete.add(j)
    
    return positions, velocities, radii, mass, to_delete

def remove_merged_particles(positions, velocities, radii, mass, to_delete):
    """Drop indices listed in to_delete."""
    if not to_delete:
        return positions, velocities, radii, mass
    
    keep_indices = [i for i in range(len(positions)) if i not in to_delete]
    
    positions = positions[keep_indices]
    velocities = velocities[keep_indices]
    radii = radii[keep_indices]
    mass = mass[keep_indices]
    
    return positions, velocities, radii, mass

def apply_agglomeration(positions, velocities, radii, mass):
    """One agglomeration pass: detect overlaps, merge, prune, optional radius boost."""
    if len(positions) < 2:
        return positions, velocities, radii, mass
    
    collision_pairs = check_collisions(positions, radii)
    
    if not collision_pairs:
        return positions, velocities, radii, mass
    
    positions, velocities, radii, mass, to_delete = merge_particles(
        positions, velocities, radii, mass, collision_pairs
    )
    
    positions, velocities, radii, mass = remove_merged_particles(
        positions, velocities, radii, mass, to_delete
    )
    
    min_mass = np.min(mass)
    enhancement_factors = 1.0 + 0.5 * (mass / min_mass - 1.0)
    radii = radii * enhancement_factors
    
    return positions, velocities, radii, mass
