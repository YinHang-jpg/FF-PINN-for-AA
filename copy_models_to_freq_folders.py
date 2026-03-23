"""
自动将训练好的PINN模型复制到各个频率文件夹
运行此脚本前，请确保已经完成了基础模型和统一模型的训练
"""
import shutil
from pathlib import Path

# 定义源目录和目标频率
SOURCE_DIR = Path("PINN")
FREQUENCIES = [8, 10, 12, 14, 16, 18, 20, 22]  # kHz

# 需要复制的模型文件
MODEL_FILES = [
    "arf_model_x.pth",
    "arf_model_t.pth",
    "stokes_model_x.pth",
    "stokes_model_t.pth",
    "stokes_model_v.pth",
    "unified_model.pth",
    "unified_model_normalization_params.json",
]

def main():
    print("=" * 70)
    print("自动复制PINN模型到各频率文件夹")
    print("=" * 70)
    
    # 检查源文件是否存在
    missing_files = []
    for fname in MODEL_FILES:
        src_file = SOURCE_DIR / fname
        if not src_file.exists():
            missing_files.append(fname)
    
    if missing_files:
        print("\n❌ 错误：以下模型文件缺失，请先训练这些模型：")
        for fname in missing_files:
            print(f"   - {fname}")
        print("\n请先运行以下训练脚本：")
        print("  1. python mechanisms/ARF_PINN_x.py")
        print("  2. python mechanisms/ARF_PINN_t.py")
        print("  3. python mechanisms/STOKES_PINN_x.py")
        print("  4. python mechanisms/STOKES_PINN_t.py")
        print("  5. python mechanisms/STOKES_PINN_v.py")
        print("  6. python mechanisms/UNIFIED_PINN.py")
        return
    
    print("\n✅ 所有源模型文件存在\n")
    
    # 开始复制
    success_count = 0
    for freq_k in FREQUENCIES:
        freq_dir = SOURCE_DIR / f"freq_{freq_k}k"
        
        # 创建目录（如果不存在）
        freq_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n处理频率 {freq_k}kHz...")
        for fname in MODEL_FILES:
            src_file = SOURCE_DIR / fname
            dst_file = freq_dir / fname
            
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  ✓ 复制: {fname}")
            except Exception as e:
                print(f"  ✗ 失败: {fname} - {e}")
        
        success_count += 1
    
    print("\n" + "=" * 70)
    print(f"✅ 完成！已处理 {success_count}/{len(FREQUENCIES)} 个频率文件夹")
    print("=" * 70)
    print("\n现在你可以运行 kernel_extractor.py 了！")

if __name__ == "__main__":
    main()
