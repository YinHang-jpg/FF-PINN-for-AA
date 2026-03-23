"""
测试频率检测是否正确工作
"""
import os
import sys

# 测试不同的PINN_FREQ_FOLDER设置
test_cases = [
    ('freq_8k', 8000),
    ('freq_10k', 10000),
    ('freq_14k', 14000),
    ('freq_22k', 22000),
]

print("测试频率检测功能\n" + "="*60)

for freq_folder, expected_freq in test_cases:
    # 设置环境变量
    os.environ['PINN_FREQ_FOLDER'] = freq_folder
    
    # 模拟clustering.py中的频率检测逻辑
    import re
    freq_match = re.search(r'freq_(\d+)k', freq_folder)
    if freq_match:
        actual_frequency = int(freq_match.group(1)) * 1000
        status = "✓" if actual_frequency == expected_freq else "✗"
        print(f"{status} {freq_folder} -> {actual_frequency} Hz (期望: {expected_freq} Hz)")
    else:
        print(f"✗ {freq_folder} -> 无法提取频率")

print("\n" + "="*60)
print("测试完成！")
