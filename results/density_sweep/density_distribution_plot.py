"""
Automate density-sweep PINN particle simulations
Features:
1. Fixed frequency: 10kHz
2. Sweep particle density from 1000 to 5000 kg/m^3 in steps of 1000
3. For each density, train five PINN sub-models into a density-specific folder
4. Skip training when checkpoints already exist
5. Run the simulation and save result figures
"""
 
import os
import sys
import shutil
import subprocess
import time
from pathlib import Path
import re
 
# Repo root on sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
 
# Density range: 1000 to 5000 kg/m^3, step 1000
densities = range(1000, 6000, 1000)  # [1000, 2000, 3000, 4000, 5000]
 
# Fixed frequency: 10kHz
FIXED_FREQUENCY = 10000
 
# File paths
sound_source_file = project_root / "initialization" / "sound_source_standing.py"
particle_init_file = project_root / "initialization" / "particle_initialization.py"
main_script = project_root / "other_files" / "PINN_main_integrated.py"
output_dir = Path(__file__).parent  # results/density_sweep
pinn_base_dir = project_root / "PINN" / "density"
 
# Training scripts
training_scripts = [
    project_root / "mechanisms" / "ARF_PINN_x.py",
    project_root / "mechanisms" / "ARF_PINN_t.py",
    project_root / "mechanisms" / "STOKES_PINN_x.py",
    project_root / "mechanisms" / "STOKES_PINN_t.py",
    project_root / "mechanisms" / "STOKES_PINN_v.py",
]
 
# Artifacts produced by each training script
model_files = {
    "ARF_PINN_x.py": ["arf_model_x.pth", "arf_model_x_normalization_params.json"],
    "ARF_PINN_t.py": ["arf_model_t.pth", "arf_model_t_normalization_params.json"],
    "STOKES_PINN_x.py": ["stokes_model_x.pth", "stokes_model_x_normalization_params.json"],
    "STOKES_PINN_t.py": ["stokes_model_t.pth", "stokes_model_t_normalization_params.json"],
    "STOKES_PINN_v.py": ["stokes_model_v.pth", "stokes_model_v_normalization_params.json"],
}
 
def backup_file(file_path):
    """Backup a file"""
    backup_path = file_path.with_suffix('.py.backup')
    shutil.copy2(file_path, backup_path)
    return backup_path
 
def restore_file(file_path, backup_path):
    """Restore a file"""
    if backup_path.exists():
        shutil.copy2(backup_path, file_path)
        backup_path.unlink()
 
def modify_frequency_in_file(file_path, freq):
    """Patch top-level frequency assignment in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
   
    modified = False
    for i, line in enumerate(lines):
        # Skip comments and function definitions
        if line.strip().startswith('#') or 'def ' in line:
            continue
       
        # Match top-level assignments only
        if '(' not in line and line.strip().startswith('frequency'):
            pattern = r'^(\s*frequency\s*=\s*)\d+(.*)$'
            match = re.match(pattern, line)
            if match:
                lines[i] = f"{match.group(1)}{freq}{match.group(2)}\n"
                modified = True
                print(f"     {file_path.name} frequency {freq} Hz")
                break
   
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
   
    return modified
 
def modify_density_in_file(file_path, density, param_name='density'):
    """
   
    Args:
        file_path: File paths
        density: 
        param_name: , 'density''particle_density'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
   
    #
    #
    lines = content.split('\n')
    modified_lines = []
    modified = False
   
    for i, line in enumerate(lines):
        #
        if line.strip().startswith('#'):
            modified_lines.append(line)
            continue
       
        #
        if 'def ' in line and f'{param_name}=' in line:
            #
            pattern = rf'({param_name}\s*=\s*)\d+'
            if re.search(pattern, line):
                new_line = re.sub(pattern, rf'\g<1>{density}', line)
                modified_lines.append(new_line)
                modified = True
                print(f"     {file_path.name}  {param_name}  {density}")
                continue
       
        #
        if line.strip().startswith(param_name) and '=' in line and 'def ' not in line:
            pattern = rf'^(\s*{param_name}\s*=\s*)\d+(.*)$'
            match = re.match(pattern, line)
            if match:
                new_line = f"{match.group(1)}{density}{match.group(2)}"
                modified_lines.append(new_line)
                modified = True
                print(f"     {file_path.name}  {param_name}  {density}")
                continue
       
        modified_lines.append(line)
   
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))
   
    return modified
 
def check_models_exist(density_folder_path):
    """"""
    required_files = []
    for files in model_files.values():
        required_files.extend(files)
   
    for file in required_files:
        if not (density_folder_path / file).exists():
            return False
    return True
 
def train_model(script_path, density, density_folder):
    """PINN"""
    script_name = script_path.name
    print(f"\n  : {script_name}")
   
    try:
        #
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
       
        # Repo root on PYTHONPATH
        project_root_abs = str(project_root.absolute())
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = project_root_abs + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = project_root_abs
       
        #
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path.absolute())],
            cwd=str(project_root.absolute()),
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        elapsed_time = time.time() - start_time
       
        if result.returncode != 0:
            print(f"    ✗  ({elapsed_time:.1f}s)")
            print(f"    : {result.stderr[-500:]}")
            return False
       
        print(f"    ✓  ({elapsed_time:.1f}s)")
       
        #
        files_to_move = model_files.get(script_name, [])
        density_folder_path = pinn_base_dir / density_folder
        for file in files_to_move:
            src = project_root / "PINN" / file
            dst = density_folder_path / file
            if src.exists():
                shutil.move(str(src), str(dst))
                print(f"    : {file}")
       
        return True
       
    except Exception as e:
        print(f"    ✗ : {str(e)}")
        return False
 
def train_all_models_for_density(density):
    """"""
    density_folder = f"density_{density}"
    density_folder_path = pinn_base_dir / density_folder
   
    print(f"\n{'='*70}")
    print(f" {density} kg/m³ ")
    print(f": {density_folder_path.absolute()}")
    print(f"{'='*70}")
   
    #
    density_folder_path.mkdir(parents=True, exist_ok=True)
   
    #
    if check_models_exist(density_folder_path):
        print(f"✓  {density_folder},")
        return True
   
    print(f" 5  PINN ...")
   
    #
    files_to_modify = [
        (particle_init_file, 'density'),
        (main_script, 'particle_density'),
    ]
   
    #
    files_to_fix_freq = [
        sound_source_file,
        project_root / "mechanisms" / "ARF.py",
        project_root / "mechanisms" / "Stokes_drag.py",
        project_root / "mechanisms" / "STOKES_PINN_x.py",
        project_root / "mechanisms" / "STOKES_PINN_t.py",
        project_root / "mechanisms" / "STOKES_PINN_v.py",
    ]
   
    #
    backups = {}
   
    print(f"\n...")
   
    #
    for file_path, param_name in files_to_modify:
        if file_path.exists():
            backup_path = backup_file(file_path)
            backups[file_path] = backup_path
            modify_density_in_file(file_path, density, param_name)
   
    #
    for file_path in files_to_fix_freq:
        if file_path.exists():
            if file_path not in backups:
                backup_path = backup_file(file_path)
                backups[file_path] = backup_path
            modify_frequency_in_file(file_path, FIXED_FREQUENCY)
   
    success_count = 0
    try:
        #
        for script_path in training_scripts:
            if train_model(script_path, density, density_folder):
                success_count += 1
            else:
                print(f"  : {script_path.name} ")
    finally:
        #
        print(f"\n...")
        for original_path, backup_path in backups.items():
            restore_file(original_path, backup_path)
            print(f"  : {original_path.name}")
   
    print(f"\n: {success_count}/{len(training_scripts)} ")
   
    #
    if check_models_exist(density_folder_path):
        print(f"✓ ")
        return True
    else:
        print(f"✗ ")
        return False
 
def generate_density_plot(initial_positions, final_positions, x_grid, relative_change, density, output_path):
    """()"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    #
    ax.plot(x_grid, relative_change, '-', color='#1f77b4', linewidth=2.5, 
            label='Relative Concentration Change', alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlim(2.0, 32.0)
    ax.set_xlabel('Position x (mm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Concentration Change (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Particle Density Distribution (Density={density} kg/m³, dt=1μs)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    #
    if len(relative_change) > 0 and np.any(np.isfinite(relative_change)):
        y_min = float(np.nanmin(relative_change))
        y_max = float(np.nanmax(relative_change))
        y_range = y_max - y_min
        y_margin = max(y_range * 0.15, 0.5)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f": {output_path}")
    
    #
    from pathlib import Path
    density_curve_file = Path(output_path).parent / f"density_curve_density_{density}.txt"
    
    with open(density_curve_file, 'w') as f:
        for x, y in zip(x_grid, relative_change):
            f.write(f"{x:.18e} {y:.18e}\n")
    
    print(f": {density_curve_file}")

def run_simulation(density):
    """()PINN"""
    density_folder = f"density_{density}"
    density_folder_path = pinn_base_dir / density_folder
   
    print(f"\n{'='*70}")
    print(f": {density} kg/m³")
    print(f"{'='*70}\n")
   
    #
    if not train_all_models_for_density(density):
        print(f":")
        return False
   
    #
    if not check_models_exist(density_folder_path):
        print(f":")
        return False
   
    print(f"\n{'='*70}")
    print(f":  {density} kg/m³")
    print(f": PINN/density/{density_folder}")
    print(f"{'='*70}\n")
   
    try:
        #
        original_pinn_folder = os.environ.get('PINN_DENSITY_FOLDER', None)
        os.environ['PINN_DENSITY_FOLDER'] = density_folder
        
        try:
            #
            import importlib
            if 'PINN_main_integrated' not in sys.modules:
                sys.path.insert(0, str(project_root))
                import PINN_main_integrated
            else:
                import PINN_main_integrated
                #
                importlib.reload(PINN_main_integrated)
            
            #
            start_time = time.time()
            print(f" run_single_simulation() (density={density} kg/m³)...")
            
            x_grid, relative_change = PINN_main_integrated.run_single_simulation(
                total_steps=10000,
                suppress_output=False,
                save_positions=False,
                particle_density=density  # 
            )
            
            elapsed_time = time.time() - start_time
            print(f"\n,: {elapsed_time:.2f} ")
            
            #
            output_image = output_dir / f"particle_distribution_density_{density}.png"
            generate_density_plot(None, None, x_grid, relative_change, density, output_image)
            
        finally:
            #
            if original_pinn_folder is not None:
                os.environ['PINN_DENSITY_FOLDER'] = original_pinn_folder
            elif 'PINN_DENSITY_FOLDER' in os.environ:
                del os.environ['PINN_DENSITY_FOLDER']
       
        return True
       
    except Exception as e:
        print(f"\n: - {str(e)}")
        import traceback
        traceback.print_exc()
        return False
 
def create_summary_csv():
    """CSV"""
    import csv
   
    csv_file = output_dir / "density_sweep_results.csv"
   
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Density (kg/m³)', 'Image File', 'Density Curve File'])
       
        for density in densities:
            image_file = f"particle_distribution_density_{density}.png"
            density_curve_file = f"density_curve_density_{density}.txt"
           
            #
            if (output_dir / image_file).exists():
                writer.writerow([density, image_file, density_curve_file])
   
    print(f"\n: {csv_file}")
 
def main():
    """"""
    print("="*70)
    print("PINN()")
    print("="*70)
    print(f"frequency: {FIXED_FREQUENCY} Hz ({FIXED_FREQUENCY/1000:.0f} kHz)")
    print(f": {min(densities)} - {max(densities)} kg/m³")
    print(f": {densities.step} kg/m³")
    print(f"PINN: {pinn_base_dir}")
    print(f": {output_dir}")
    print(f": {len(densities)} ")
    print(f"\n:")
    print(f"  1. / 5  PINN ()")
    print(f"  2. ")
    print(f"  3. \n")
   
    #
    output_dir.mkdir(parents=True, exist_ok=True)
   
    success_count = 0
    failed_densities = []
   
    try:
        #
        for i, density in enumerate(densities, 1):
            print(f"\n: {i}/{len(densities)}")
           
            if run_simulation(density):
                success_count += 1
            else:
                failed_densities.append(density)
           
            #
            import gc
            gc.collect()
           
            #
            if i < len(densities):
                time.sleep(2)
       
        #
        create_summary_csv()
       
        #
        print("\n" + "="*70)
        print("！")
        print("="*70)
        print(f": {success_count}/{len(densities)}")
        if failed_densities:
            print(f": {failed_densities} kg/m³")
        print(f": {output_dir}")
       
    except KeyboardInterrupt:
        print("\n\n")
   
    finally:
        print("\n")
 
if __name__ == "__main__":
    main()
 
 
 
 
