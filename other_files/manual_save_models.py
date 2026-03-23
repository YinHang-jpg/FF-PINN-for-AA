"""
手动保存当前PINN根目录的模型到指定频率文件夹
用于救急：当训练完成但模型还在根目录时使用
"""
import shutil
from pathlib import Path
import json

project_root = Path(__file__).parent.parent.parent

# 读取当前频率
freq_file = project_root / 'initialization' / 'sound_source_standing.py'
with open(freq_file, 'r') as f:
    for line in f:
        if 'frequency =' in line and not line.strip().startswith('#'):
            import re
            match = re.search(r'frequency\s*=\s*(\d+)', line)
            if match:
                current_frequency = int(match.group(1))
                break

print(f"当前配置频率: {current_frequency} Hz ({current_frequency//1000} kHz)")

# 确认
answer = input(f"\n将 PINN/ 根目录的模型保存到 freq_{current_frequency//1000}k/ 吗? (yes/no): ")
if answer.lower() != 'yes':
    print("已取消")
    exit(0)

# 创建目标文件夹
freq_k = current_frequency // 1000
freq_folder = project_root / 'PINN' / f'freq_{freq_k}k'
freq_folder.mkdir(parents=True, exist_ok=True)

# 要复制的文件
files_to_copy = [
    'arf_model_x.pth',
    'arf_model_x_normalization_params.json',
    'arf_model_t.pth',
    'arf_model_t_normalization_params.json',
    'stokes_model_x.pth',
    'stokes_model_x_normalization_params.json',
    'stokes_model_v.pth',
    'stokes_model_v_normalization_params.json',
    'stokes_model_t.pth',
    'stokes_model_t_normalization_params.json',
]

pinn_root = project_root / 'PINN'

copied = []
missing = []

for filename in files_to_copy:
    src = pinn_root / filename
    dst = freq_folder / filename
    
    if src.exists():
        shutil.copy2(src, dst)
        copied.append(filename)
        print(f"✓ 复制: {filename}")
    else:
        missing.append(filename)
        print(f"✗ 缺失: {filename}")

print(f"\n完成！")
print(f"  成功复制: {len(copied)} 个文件")
print(f"  缺失: {len(missing)} 个文件")
print(f"  保存到: {freq_folder.relative_to(project_root)}")

# 验证归一化参数
norm_file = freq_folder / 'arf_model_t_normalization_params.json'
if norm_file.exists():
    with open(norm_file, 'r') as f:
        data = json.load(f)
    t_max = data.get('t_max', 0)
    if t_max > 0:
        inferred_freq = 1.0 / t_max
        print(f"\n验证: t_max={t_max:.6f}s → 频率={inferred_freq:.0f} Hz")
        if abs(inferred_freq - current_frequency) < 1:
            print("✓ 频率匹配！")
        else:
            print(f"⚠ 警告: 频率不匹配！期望 {current_frequency} Hz, 实际 {inferred_freq:.0f} Hz")
