import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_comsol_curve():
    raw = pd.read_csv('validation/comsol_positions.csv', header=None, skip_blank_lines=True)
    raw = raw.dropna(axis=1, how='all')
    raw = raw.apply(pd.to_numeric, errors='coerce')
    raw = raw.dropna(axis=0, subset=[0])
    positions_df = raw.iloc[:, 1:]
    positions_df = positions_df.dropna(axis=1, how='all')
    particle_positions = positions_df.to_numpy()
    if particle_positions.shape[0] < 2:
        return np.zeros(100), np.zeros(100)
    x_grid = np.linspace(0.0, 34.0, 100)
    x0 = particle_positions[0, :]
    xT = particle_positions[-1, :]
    x0 = x0[np.isfinite(x0)]
    xT = xT[np.isfinite(xT)]
    def gauss_density(xx, grid):
        sigma=0.2
        c = np.zeros_like(grid)
        for i, g in enumerate(grid):
            distances = np.abs(xx * 1000.0 - g)
            c[i]=np.sum(np.exp(-distances**2/(2*sigma**2)))
        return c
    initial = gauss_density(x0, x_grid)
    final = gauss_density(xT, x_grid)
    with np.errstate(divide='ignore', invalid='ignore'):
        rc = (final-initial)/initial*100.0
        rc = np.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0)
    return x_grid, rc

def run_and_get_curve_from_txt(pyfile, txtfile):
    if os.path.exists(txtfile):
        os.remove(txtfile)
    ret = subprocess.run(['python', pyfile], timeout=600)
    arr = np.loadtxt(txtfile)
    if arr.shape[1] == 2:
        return arr[:,0], arr[:,1]
    return arr[0], arr[1]

def main():
    x1, y1 = get_comsol_curve()
    x2, y2 = run_and_get_curve_from_txt('DEM_main.py', 'density_curve_DEM.txt')
    x3, y3 = run_and_get_curve_from_txt('PINN_main_integrated.py', 'density_curve_PINN.txt')
    plt.figure(figsize=(12, 7))
    plt.plot(x1, y1, label="COMSOL (baseline)", color='black', linewidth=2)
    plt.plot(x2, y2, label="DEM", color='blue')
    plt.plot(x3, y3, label="PINN", color='red')
    plt.xlabel("x (mm)")
    plt.ylabel("Relative Change (%)")
    plt.title("Particle Density Distribution Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(5, 30)
    plt.tight_layout()
    # 误差曲线
    err1 = y2-y1
    err2 = y3-y1
    idx = np.where((x1 >= 5) & (x1 <= 30))[0]
    if idx.size > 1:
        data_min = min(np.nanmin(err1[idx]), np.nanmin(err2[idx]))
        data_max = max(np.nanmax(err1[idx]), np.nanmax(err2[idx]))
    else:
        data_min = min(np.nanmin(err1), np.nanmin(err2))
        data_max = max(np.nanmax(err1), np.nanmax(err2))
    ymin = data_min*1.1 if data_min < 0 else data_min*0.9
    ymax = data_max*1.1 if data_max > 0 else data_max*0.9
    ymin = min(ymin, -1)
    ymax = max(ymax, 1)
    plt.figure(figsize=(12, 7))
    plt.plot(x1, err1, label="DEM - COMSOL", color='blue')
    plt.plot(x1, err2, label="PINN - COMSOL", color='red')
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero Baseline')
    plt.xlabel("x (mm)")
    plt.ylabel("Error in Relative Change (%)")
    plt.title("Deviation from COMSOL Baseline")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(5, 30)
    plt.ylim(ymin, ymax)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


