import os
import re
import csv
import time
import shutil
import importlib
import io
import contextlib
from PIL import Image

# Non-interactive backend for automation
import matplotlib
matplotlib.use('Agg')


def run_single_experiment(particle_count: int):
    """
    Run ``clustering.main()`` once with a patched particle count; crop/rename the 2x2 figure.

    Returns (total_collisions, elapsed_seconds, figure_path_or_None).
    """
    import clustering  # noqa: WPS433

    original_init = clustering.initialize_particles

    def _patched_initialize_particles(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs['N'] = particle_count
        return original_init(*args, **kwargs)

    clustering.initialize_particles = _patched_initialize_particles  # type: ignore[attr-defined]

    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try:
            clustering.main()
        except SystemExit:
            pass
    end = time.perf_counter()
    elapsed = end - start
    output = buf.getvalue()

    collisions = None
    m = re.search(r'Total collisions:\s*([0-9]+)', output)
    if m:
        try:
            collisions = int(m.group(1))
        except Exception:
            collisions = None

    if collisions is None:
        collisions = -1

    # Keep collision-trend panel (bottom-right of clustering_results.png)
    src_png = 'clustering_results.png'
    dst_png = f'collision_trend_N_{particle_count}.png'
    figure_path = None
    if os.path.exists(src_png):
        try:
            # Crop bottom-right quadrant
            with Image.open(src_png) as img:
                w, h = img.size
                box = (w // 2, h // 2, w, h)  # (left, top, right, bottom)
                cropped = img.crop(box)
                cropped.save(dst_png)
            figure_path = os.path.abspath(dst_png)
            # Remove full composite to avoid confusion
            try:
                os.remove(src_png)
            except Exception:
                pass
        except Exception:
            # Fallback: rename whole figure
            try:
                shutil.move(src_png, dst_png)
                figure_path = os.path.abspath(dst_png)
            except Exception:
                figure_path = None

    return collisions, elapsed, figure_path


def main():
    particle_counts = list(range(1000, 15001, 1000))

    results = []

    import clustering  # noqa: WPS433

    for idx, n in enumerate(particle_counts):
        if idx > 0:
            clustering = importlib.reload(clustering)  # type: ignore[F811]

        collisions, elapsed, figure_path = run_single_experiment(n)
        results.append((n, collisions, elapsed, figure_path))
        print(f'[auto] N={n}, collisions={collisions}, elapsed={elapsed:.3f}s, figure={figure_path or "N/A"}')

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


