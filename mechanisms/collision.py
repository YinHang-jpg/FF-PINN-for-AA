import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, List, Optional

def detect_collisions(positions: np.ndarray, radii: np.ndarray) -> List[Tuple[int, int]]:
    """
    检测粒子之间的碰撞
    
    Args:
        positions: 粒子位置数组 (N, 2)
        radii: 粒子半径数组 (N,)
    
    Returns:
        碰撞粒子对的索引列表 [(i, j), ...]
    """
    collisions = []
    N = len(positions)
    
    # 使用KDTree进行快速近邻搜索
    tree = cKDTree(positions)
    
    for i in range(N):
        # 查找距离小于等于两粒子半径之和的粒子
        collision_radius = radii[i] + radii.max()  # 最大可能的碰撞距离
        neighbors = tree.query_ball_point(positions[i], collision_radius)
        
        for j in neighbors:
            if i < j:  # 避免重复检测
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= (radii[i] + radii[j]):
                    collisions.append((i, j))
    
    return collisions

def is_head_on_collision(pos1: np.ndarray, pos2: np.ndarray, 
                        vel1: np.ndarray, vel2: np.ndarray, 
                        radius1: float, radius2: float) -> bool:
    """
    判断是否为对心碰撞
    
    Args:
        pos1, pos2: 两个粒子的位置
        vel1, vel2: 两个粒子的速度
        radius1, radius2: 两个粒子的半径
    
    Returns:
        是否为对心碰撞
    """
    # 计算相对位置向量
    rel_pos = pos2 - pos1
    distance = np.linalg.norm(rel_pos)
    
    # 如果粒子已经重叠，认为是碰撞
    if distance <= (radius1 + radius2):
        return True
    
    # 计算相对速度
    rel_vel = vel2 - vel1
    
    # 计算相对位置和相对速度的点积
    # 如果为负，说明粒子正在接近
    dot_product = np.dot(rel_pos, rel_vel)
    
    # 计算碰撞时间（如果粒子会碰撞）
    if dot_product < 0:  # 粒子正在接近
        # 计算碰撞时间
        rel_vel_mag = np.linalg.norm(rel_vel)
        if rel_vel_mag > 1e-10:  # 避免除零
            collision_time = (distance - radius1 - radius2) / rel_vel_mag
            
            # 如果碰撞时间很短（下一帧内），认为是碰撞
            if collision_time <= 0.001:  # 1ms内
                return True
    
    return False

def calculate_collision_velocity(vel1: np.ndarray, vel2: np.ndarray, 
                               mass1: float, mass2: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算完全非弹性碰撞后的速度
    
    Args:
        vel1, vel2: 碰撞前的速度
        mass1, mass2: 粒子质量
    
    Returns:
        碰撞后的速度 (vel1_new, vel2_new)
    """
    # 完全非弹性碰撞：碰撞后两粒子速度相同
    total_mass = mass1 + mass2
    momentum = mass1 * vel1 + mass2 * vel2
    
    # 碰撞后的共同速度
    common_velocity = momentum / total_mass
    
    return common_velocity, common_velocity

def merge_particles(positions: np.ndarray, velocities: np.ndarray, 
                   radii: np.ndarray, mass: np.ndarray,
                   collision_pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    合并碰撞的粒子
    
    Args:
        positions: 粒子位置数组
        velocities: 粒子速度数组
        radii: 粒子半径数组
        mass: 粒子质量数组
        collision_pairs: 碰撞粒子对列表
    
    Returns:
        更新后的位置、速度、半径和质量数组
    """
    if not collision_pairs:
        return positions, velocities, radii, mass
    
    # 按质量从大到小排序碰撞对，确保大粒子优先
    collision_pairs.sort(key=lambda pair: mass[pair[0]] + mass[pair[1]], reverse=True)
    
    # 记录要删除的粒子索引
    particles_to_remove = set()
    
    # 处理每个碰撞对
    for i, j in collision_pairs:
        if i in particles_to_remove or j in particles_to_remove:
            continue  # 跳过已经被处理的粒子
        
        # 计算合并后的属性
        total_mass = mass[i] + mass[j]
        
        # 质量加权平均位置
        merged_position = (mass[i] * positions[i] + mass[j] * positions[j]) / total_mass
        
        # 动量守恒的速度
        merged_velocity = (mass[i] * velocities[i] + mass[j] * velocities[j]) / total_mass
        
        # 体积守恒的半径（假设粒子为球形）
        merged_radius = np.cbrt(radii[i]**3 + radii[j]**3)
        
        # 更新第一个粒子的属性
        positions[i] = merged_position
        velocities[i] = merged_velocity
        radii[i] = merged_radius
        mass[i] = total_mass
        
        # 标记第二个粒子为待删除
        particles_to_remove.add(j)
    
    # 删除已合并的粒子
    if particles_to_remove:
        keep_indices = [i for i in range(len(positions)) if i not in particles_to_remove]
        
        positions = positions[keep_indices]
        velocities = velocities[keep_indices]
        radii = radii[keep_indices]
        mass = mass[keep_indices]
    
    return positions, velocities, radii, mass

def apply_collision_physics(positions: np.ndarray, velocities: np.ndarray, 
                           radii: np.ndarray, mass: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    应用碰撞物理
    
    Args:
        positions: 粒子位置数组
        velocities: 粒子速度数组
        radii: 粒子半径数组
        mass: 粒子质量数组
        dt: 时间步长
    
    Returns:
        更新后的位置、速度、半径和质量数组
    """
    if len(positions) < 2:
        return positions, velocities, radii, mass
    
    # 检测碰撞
    collision_pairs = detect_collisions(positions, radii)
    
    # 过滤出对心碰撞
    head_on_collisions = []
    for i, j in collision_pairs:
        if is_head_on_collision(positions[i], positions[j], 
                               velocities[i], velocities[j], 
                               radii[i], radii[j]):
            head_on_collisions.append((i, j))
    
    # 处理碰撞
    if head_on_collisions:
        # 先更新碰撞后的速度
        for i, j in head_on_collisions:
            vel1_new, vel2_new = calculate_collision_velocity(
                velocities[i], velocities[j], mass[i], mass[j]
            )
            velocities[i] = vel1_new
            velocities[j] = vel2_new
        
        # 然后合并粒子
        positions, velocities, radii, mass = merge_particles(
            positions, velocities, radii, mass, head_on_collisions
        )
    
    return positions, velocities, radii, mass

def calculate_collision_energy(vel1: np.ndarray, vel2: np.ndarray, 
                              mass1: float, mass2: float) -> float:
    """
    计算碰撞前的动能
    
    Args:
        vel1, vel2: 碰撞前的速度
        mass1, mass2: 粒子质量
    
    Returns:
        碰撞前的总动能
    """
    ke1 = 0.5 * mass1 * np.linalg.norm(vel1)**2
    ke2 = 0.5 * mass2 * np.linalg.norm(vel2)**2
    return ke1 + ke2

def calculate_merged_particle_properties(positions: np.ndarray, velocities: np.ndarray, 
                                       radii: np.ndarray, mass: np.ndarray,
                                       collision_pairs: List[Tuple[int, int]]) -> dict:
    """
    计算合并粒子的属性统计
    
    Args:
        positions: 粒子位置数组
        velocities: np.ndarray, radii: np.ndarray, mass: np.ndarray
        collision_pairs: 碰撞粒子对列表
    
    Returns:
        包含统计信息的字典
    """
    if not collision_pairs:
        return {}
    
    stats = {
        'total_collisions': len(collision_pairs),
        'collision_pairs': collision_pairs,
        'energy_loss': 0.0,
        'merged_particles': []
    }
    
    for i, j in collision_pairs:
        # 计算碰撞前的动能
        initial_ke = calculate_collision_energy(velocities[i], velocities[j], mass[i], mass[j])
        
        # 计算合并后的动能
        merged_vel = (mass[i] * velocities[i] + mass[j] * velocities[j]) / (mass[i] + mass[j])
        merged_mass = mass[i] + mass[j]
        final_ke = 0.5 * merged_mass * np.linalg.norm(merged_vel)**2
        
        # 能量损失
        energy_loss = initial_ke - final_ke
        stats['energy_loss'] += energy_loss
        
        # 记录合并信息
        stats['merged_particles'].append({
            'particle1': {'index': i, 'mass': mass[i], 'radius': radii[i]},
            'particle2': {'index': j, 'mass': mass[j], 'radius': radii[j]},
            'merged_mass': merged_mass,
            'merged_radius': np.cbrt(radii[i]**3 + radii[j]**3),
            'energy_loss': energy_loss
        })
    
    return stats
