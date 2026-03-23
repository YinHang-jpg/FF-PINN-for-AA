import os
import re
import csv
import time
import shutil
import importlib
import io
import contextlib
from PIL import Image

# 使用非交互式后端，避免弹窗干扰自动化流程
import matplotlib
matplotlib.use('Agg')


def run_single_experiment(particle_count: int):
    """
    运行一次 clustering.main()，强制使用给定的粒子数，返回统计结果并重命名输出图像。
    返回: (total_collisions:int, elapsed_seconds:float, figure_path:str or None)
    """
    # 延迟导入，便于每次迭代重载模块
    import clustering  # noqa: WPS433

    # 备份原始的 initialize_particles
    original_init = clustering.initialize_particles

    # 定义包装器：忽略传入的 N，使用我们期望的粒子数
    def _patched_initialize_particles(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['N'] = particle_count
        return original_init(*args, **kwargs)

    # 猴补模块作用域内的函数
    clustering.initialize_particles = _patched_initialize_particles  # type: ignore[attr-defined]

    # 捕获 stdout，解析统计信息
    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try:
            clustering.main()
        except SystemExit:
            # 防御性处理：若主程序调用了 sys.exit
            pass
    end = time.perf_counter()
    elapsed = end - start
    output = buf.getvalue()

    # 解析总碰撞次数（形如：总碰撞次数: 1234）
    collisions = None
    m = re.search(r'总碰撞次数:\s*([0-9]+)', output)
    if m:
        try:
            collisions = int(m.group(1))
        except Exception:
            collisions = None

    # 回退：若未解析到，设为 -1 便于排查
    if collisions is None:
        collisions = -1

    # 处理并仅保存“碰撞趋势”子图（clustering 会保存 2x2 的 clustering_results.png）
    src_png = 'clustering_results.png'
    dst_png = f'collision_trend_N_{particle_count}.png'
    figure_path = None
    if os.path.exists(src_png):
        try:
            # 裁剪右下角象限（2x2子图中的第4张）
            with Image.open(src_png) as img:
                w, h = img.size
                box = (w // 2, h // 2, w, h)  # (left, top, right, bottom)
                cropped = img.crop(box)
                cropped.save(dst_png)
            figure_path = os.path.abspath(dst_png)
            # 删除原始整图，避免混淆
            try:
                os.remove(src_png)
            except Exception:
                pass
        except Exception:
            # 回退：若裁剪失败，则直接重命名整图（但仍尽量不覆盖原文件）
            try:
                shutil.move(src_png, dst_png)
                figure_path = os.path.abspath(dst_png)
            except Exception:
                figure_path = None

    return collisions, elapsed, figure_path


def main():
    # 循环粒子数：从1000到15000，步长1000
    particle_counts = list(range(1000, 15001, 1000))

    results = []  # 每项为 (N, collisions, elapsed_s, figure_path)

    # 首次导入模块
    import clustering  # noqa: WPS433

    for idx, n in enumerate(particle_counts):
        # 每轮前重载模块，确保状态干净
        if idx > 0:
            clustering = importlib.reload(clustering)  # type: ignore[F811]

        collisions, elapsed, figure_path = run_single_experiment(n)
        results.append((n, collisions, elapsed, figure_path))
        print(f'[auto] N={n}, collisions={collisions}, elapsed={elapsed:.3f}s, figure={figure_path or "N/A"}')

    # 写出 CSV 汇总
    csv_path = 'auto_test_results.csv'
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['particles', 'total_collisions', 'elapsed_seconds', 'figure_path'])
            for n, c, t, p in results:
                writer.writerow([n, c, f'{t:.6f}', p or ''])
        print(f'[auto] Results saved to: {os.path.abspath(csv_path)}')
    except Exception as e:
        print(f'[auto] Failed to write CSV: {e}')


if __name__ == '__main__':
    main()


