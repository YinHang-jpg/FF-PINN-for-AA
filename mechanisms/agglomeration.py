import numpy as np

def check_collisions(positions, radii):
    """
    检查粒子之间的碰撞
    :param positions: 粒子位置数组 (N, 2)
    :param radii: 粒子半径数组 (N,)
    :return: 碰撞对列表 [(i, j), ...]
    """
    N = len(positions)
    collisions = []
    
    for i in range(N):
        for j in range(i + 1, N):
            # 计算两粒子中心距离
            distance = np.linalg.norm(positions[i] - positions[j])
            # 如果距离小于两半径之和，则发生碰撞
            if distance < (radii[i] + radii[j]):
                collisions.append((i, j))
    
    return collisions

def merge_particles(positions, velocities, radii, mass, collision_pairs):
    """
    合并碰撞的粒子（完全非弹性碰撞）
    :param positions: 粒子位置数组
    :param velocities: 粒子速度数组
    :param radii: 粒子半径数组
    :param mass: 粒子质量数组
    :param collision_pairs: 碰撞对列表
    :return: 更新后的位置、速度、半径、质量数组，以及要删除的粒子索引
    """
    N = len(positions)
    to_delete = set()
    
    # 处理每个碰撞对
    for i, j in collision_pairs:
        if i in to_delete or j in to_delete:
            continue  # 跳过已经处理过的粒子
            
        # 计算合并后的质量（质量守恒）
        total_mass = mass[i] + mass[j]
        
        # 计算合并后的位置（质心）
        center_of_mass = (positions[i] * mass[i] + positions[j] * mass[j]) / total_mass
        
        # 计算合并后的速度（动量守恒）
        total_momentum = velocities[i] * mass[i] + velocities[j] * mass[j]
        merged_velocity = total_momentum / total_mass
        
        # 计算合并后的半径（体积守恒，假设二维为面积守恒）
        # 新半径：sqrt(r1^2 + r2^2) * 增强因子，让合并后的粒子更明显
        merged_radius = np.sqrt(radii[i]**2 + radii[j]**2) * 1.2
        
        # 更新第一个粒子的属性
        positions[i] = center_of_mass
        velocities[i] = merged_velocity
        radii[i] = merged_radius
        mass[i] = total_mass
        
        # 标记第二个粒子为删除
        to_delete.add(j)
    
    return positions, velocities, radii, mass, to_delete

def remove_merged_particles(positions, velocities, radii, mass, to_delete):
    """
    删除已合并的粒子
    :param positions: 粒子位置数组
    :param velocities: 粒子速度数组
    :param radii: 粒子半径数组
    :param mass: 粒子质量数组
    :param to_delete: 要删除的粒子索引集合
    :return: 清理后的数组
    """
    if not to_delete:
        return positions, velocities, radii, mass
    
    # 创建保留索引
    keep_indices = [i for i in range(len(positions)) if i not in to_delete]
    
    # 过滤数组
    positions = positions[keep_indices]
    velocities = velocities[keep_indices]
    radii = radii[keep_indices]
    mass = mass[keep_indices]
    
    return positions, velocities, radii, mass

def apply_agglomeration(positions, velocities, radii, mass):
    """
    应用粒子凝聚机制
    :param positions: 粒子位置数组 (N, 2)
    :param velocities: 粒子速度数组 (N, 2)
    :param radii: 粒子半径数组 (N,)
    :param mass: 粒子质量数组 (N,)
    :return: 更新后的位置、速度、半径、质量数组
    """
    if len(positions) < 2:
        return positions, velocities, radii, mass
    
    # 检查碰撞
    collision_pairs = check_collisions(positions, radii)
    
    if not collision_pairs:
        return positions, velocities, radii, mass
    
    # 合并碰撞的粒子
    positions, velocities, radii, mass, to_delete = merge_particles(
        positions, velocities, radii, mass, collision_pairs
    )
    
    # 删除已合并的粒子
    positions, velocities, radii, mass = remove_merged_particles(
        positions, velocities, radii, mass, to_delete
    )
    
    # 应用视觉增强：让所有粒子都稍微大一点，合并后的粒子更明显
    # 基于质量比例计算增强因子
    min_mass = np.min(mass)
    enhancement_factors = 1.0 + 0.5 * (mass / min_mass - 1.0)
    radii = radii * enhancement_factors
    
    return positions, velocities, radii, mass
