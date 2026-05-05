"""Copy trained PINN checkpoints from PINN/ into per-frequency subfolders.

Run after base and unified training so each freq_*k directory has a full model set.
"""
import shutil
from pathlib import Path

SOURCE_DIR = Path("PINN")
FREQUENCIES = [8, 10, 12, 14, 16, 18, 20, 22]  # kHz

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
    print("Copy PINN weights into freq_*k folders")
    print("=" * 70)

    missing_files = []
    for fname in MODEL_FILES:
        src_file = SOURCE_DIR / fname
        if not src_file.exists():
            missing_files.append(fname)

    if missing_files:
        print("\nError: missing files (train first):")
        for fname in missing_files:
            print(f"   - {fname}")
        print("\nTypical training order:")
        print("  1. python mechanisms/ARF_PINN_x.py")
        print("  2. python mechanisms/ARF_PINN_t.py")
        print("  3. python mechanisms/STOKES_PINN_x.py")
        print("  4. python mechanisms/STOKES_PINN_t.py")
        print("  5. python mechanisms/STOKES_PINN_v.py")
        print("  6. python mechanisms/UNIFIED_PINN.py")
        return

    print("\nAll listed source files found.\n")

    success_count = 0
    for freq_k in FREQUENCIES:
        freq_dir = SOURCE_DIR / f"freq_{freq_k}k"
        freq_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFrequency {freq_k} kHz ...")
        for fname in MODEL_FILES:
            src_file = SOURCE_DIR / fname
            dst_file = freq_dir / fname
            try:
                shutil.copy2(src_file, dst_file)
                print(f"  copied: {fname}")
            except Exception as e:
                print(f"  failed: {fname} - {e}")

        success_count += 1

    print("\n" + "=" * 70)
    print(f"Done. Processed {success_count}/{len(FREQUENCIES)} frequency folders.")
    print("=" * 70)
    print("\nYou can run results/kernel/kernel_extractor.py next.")


if __name__ == "__main__":
    main()
