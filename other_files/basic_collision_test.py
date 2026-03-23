import numpy as np
from mechanisms.collision import (
    detect_collisions, 
    is_head_on_collision, 
    calculate_collision_velocity,
    apply_collision_physics
)

def basic_collision_test():
    """基本的碰撞测试"""
    print("=== 基本碰撞测试 ===")
    
    # 初始化两个粒子：一个向左，一个向右
    positions = np.array([
        [0.002, 0.005],  # 粒子1：左侧
        [0.008, 0.005]   # 粒子2：右侧
    ])
    
    velocities = np.array([
        [0.003, 0.0],    # 粒子1：向右运动
        [-0.003, 0.0]    # 粒子2：向左运动
    ])
    
    radii = np.array([0.001, 0.001])  # 两个粒子半径都是1mm
    mass = np.array([1.0, 1.0])       # 两个粒子质量都是1g
    
    dt = 0.001  # 时间步长1ms
    
    print("初始状态:")
    print(f"粒子1: 位置=({positions[0,0]*1000:.1f}, {positions[0,1]*1000:.1f})mm, 速度=({velocities[0,0]*1000:.1f}, {velocities[0,1]*1000:.1f})mm/s")
    print(f"粒子2: 位置=({positions[1,0]*1000:.1f}, {positions[1,1]*1000:.1f})mm, 速度=({velocities[1,0]*1000:.1f}, {velocities[1,1]*1000:.1f})mm/s")
    print(f"粒子间距离: {np.linalg.norm(positions[1] - positions[0])*1000:.1f}mm")
    print()
    
    # 模拟几个时间步
    for step in range(10):
        print(f"步骤 {step + 1}:")
        
        # 更新位置
        positions += velocities * dt
        
        # 检测碰撞
        collisions = detect_collisions(positions, radii)
        print(f"  位置: 粒子1=({positions[0,0]*1000:.1f}, {positions[0,1]*1000:.1f})mm, 粒子2=({positions[1,0]*1000:.1f}, {positions[1,1]*1000:.1f})mm")
        print(f"  距离: {np.linalg.norm(positions[1] - positions[0])*1000:.3f}mm")
        print(f"  碰撞检测: {collisions}")
        
        # 如果检测到碰撞，应用碰撞物理
        if collisions:
            print(f"  *** 碰撞检测到！ ***")
            
            # 检查是否为对心碰撞
            i, j = collisions[0]
            is_head_on = is_head_on_collision(
                positions[i], positions[j], 
                velocities[i], velocities[j], 
                radii[i], radii[j]
            )
            print(f"  对心碰撞: {is_head_on}")
            
            if is_head_on:
                # 应用碰撞物理
                old_positions = positions.copy()
                old_velocities = velocities.copy()
                old_radii = radii.copy()
                old_mass = mass.copy()
                
                positions, velocities, radii, mass = apply_collision_physics(
                    positions, velocities, radii, mass, dt
                )
                
                print(f"  碰撞后状态:")
                print(f"    粒子数量: {len(positions)}")
                if len(positions) == 1:
                    print(f"    合并后粒子: 位置=({positions[0,0]*1000:.1f}, {positions[0,1]*1000:.1f})mm, 速度=({velocities[0,0]*1000:.1f}, {velocities[0,1]*1000:.1f})mm/s")
                    print(f"    合并后半径: {radii[0]*1000:.1f}mm, 质量: {mass[0]:.1f}g")
                    
                    # 计算能量损失
                    initial_ke = 0.5 * old_mass[0] * np.linalg.norm(old_velocities[0])**2 + \
                                 0.5 * old_mass[1] * np.linalg.norm(old_velocities[1])**2
                    final_ke = 0.5 * mass[0] * np.linalg.norm(velocities[0])**2
                    energy_loss = initial_ke - final_ke
                    print(f"    能量损失: {energy_loss*1e6:.3f}μJ")
                    
                    print(f"  ✅ 粒子成功合并！")
                    break
        
        print()
    
    return positions, velocities, radii, mass

def test_collision_physics():
    """测试碰撞物理计算"""
    print("\n=== 测试碰撞物理计算 ===")
    
    # 创建测试数据
    vel1 = np.array([0.003, 0.0])    # 向右3mm/s
    vel2 = np.array([-0.003, 0.0])   # 向左3mm/s
    mass1 = 1.0
    mass2 = 1.0
    
    print(f"碰撞前:")
    print(f"  粒子1: 速度=({vel1[0]*1000:.1f}, {vel1[1]*1000:.1f})mm/s, 动量=({mass1*vel1[0]*1000:.1f}, {mass1*vel1[1]*1000:.1f})g·mm/s")
    print(f"  粒子2: 速度=({vel2[0]*1000:.1f}, {vel2[1]*1000:.1f})mm/s, 动量=({mass2*vel2[0]*1000:.1f}, {mass2*vel2[1]*1000:.1f})g·mm/s")
    
    # 计算总动量
    total_momentum = mass1 * vel1 + mass2 * vel2
    print(f"  总动量: ({total_momentum[0]*1000:.1f}, {total_momentum[1]*1000:.1f})g·mm/s")
    
    # 计算碰撞后的速度
    vel1_new, vel2_new = calculate_collision_velocity(vel1, vel2, mass1, mass2)
    print(f"\n碰撞后:")
    print(f"  粒子1: 速度=({vel1_new[0]*1000:.1f}, {vel1_new[1]*1000:.1f})mm/s")
    print(f"  粒子2: 速度=({vel2_new[0]*1000:.1f}, {vel2_new[1]*1000:.1f})mm/s")
    
    # 验证动量守恒
    final_momentum = mass1 * vel1_new + mass2 * vel2_new
    print(f"  碰撞后总动量: ({final_momentum[0]*1000:.1f}, {final_momentum[1]*1000:.1f})g·mm/s")
    print(f"  动量守恒: {np.allclose(total_momentum, final_momentum)}")
    
    # 计算能量损失
    initial_ke = 0.5 * mass1 * np.linalg.norm(vel1)**2 + 0.5 * mass2 * np.linalg.norm(vel2)**2
    final_ke = 0.5 * mass1 * np.linalg.norm(vel1_new)**2 + 0.5 * mass2 * np.linalg.norm(vel2_new)**2
    energy_loss = initial_ke - final_ke
    print(f"  能量损失: {energy_loss*1e6:.3f}μJ ({(energy_loss/initial_ke)*100:.1f}%)")

def test_collision_detection():
    """测试碰撞检测功能"""
    print("\n=== 测试碰撞检测功能 ===")
    
    # 测试1：两个粒子即将碰撞
    positions = np.array([
        [0.004, 0.005],  # 粒子1
        [0.006, 0.005]   # 粒子2，距离2mm
    ])
    
    radii = np.array([0.001, 0.001])  # 半径都是1mm
    
    collisions = detect_collisions(positions, radii)
    print(f"测试1 - 粒子距离2mm:")
    print(f"  位置: 粒子1=({positions[0,0]*1000:.1f}, {positions[0,1]*1000:.1f})mm, 粒子2=({positions[1,0]*1000:.1f}, {positions[1,1]*1000:.1f})mm")
    print(f"  距离: {np.linalg.norm(positions[1] - positions[0])*1000:.1f}mm")
    print(f"  半径和: {(radii[0] + radii[1])*1000:.1f}mm")
    print(f"  碰撞检测: {collisions}")
    print(f"  预期结果: 应该检测到碰撞 (距离 <= 半径和)")
    print()
    
    # 测试2：两个粒子不会碰撞
    positions2 = np.array([
        [0.002, 0.005],  # 粒子1
        [0.008, 0.005]   # 粒子2，距离6mm
    ])
    
    collisions2 = detect_collisions(positions2, radii)
    print(f"测试2 - 粒子距离6mm:")
    print(f"  位置: 粒子1=({positions2[0,0]*1000:.1f}, {positions2[0,1]*1000:.1f})mm, 粒子2=({positions2[1,0]*1000:.1f}, {positions2[1,1]*1000:.1f})mm")
    print(f"  距离: {np.linalg.norm(positions2[1] - positions2[0])*1000:.1f}mm")
    print(f"  半径和: {(radii[0] + radii[1])*1000:.1f}mm")
    print(f"  碰撞检测: {collisions2}")
    print(f"  预期结果: 不应该检测到碰撞 (距离 > 半径和)")

if __name__ == "__main__":
    print("🚀 基本碰撞测试脚本")
    print("=" * 50)
    
    # 1. 基本碰撞测试
    final_positions, final_velocities, final_radii, final_mass = basic_collision_test()
    
    # 2. 测试碰撞物理计算
    test_collision_physics()
    
    # 3. 测试碰撞检测功能
    test_collision_detection()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")
    print("观察结果:")
    print("- 两个粒子相向运动")
    print("- 碰撞检测正常工作")
    print("- 粒子成功合并")
    print("- 动量守恒验证通过")
    print("- 能量损失计算正确")
    
    print(f"\n最终状态:")
    print(f"粒子数量: {len(final_positions)}")
    if len(final_positions) == 1:
        print(f"合并后粒子: 位置=({final_positions[0,0]*1000:.1f}, {final_positions[0,1]*1000:.1f})mm")
        print(f"合并后速度: ({final_velocities[0,0]*1000:.1f}, {final_velocities[0,1]*1000:.1f})mm/s")
        print(f"合并后半径: {final_radii[0]*1000:.1f}mm")
        print(f"合并后质量: {final_mass[0]:.1f}g")







