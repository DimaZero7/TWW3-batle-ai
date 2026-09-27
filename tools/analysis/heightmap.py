"""Render sampled terrain heights from a saved map CSV; never starts the game."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(source, output, bounds, step):
    min_x, max_x, min_z, max_z = bounds
    if not all(math.isfinite(v) for v in (*bounds, step)) or step <= 0:
        raise ValueError('Bounds and step must be finite; step must be positive')
    if max_x <= min_x or max_z <= min_z:
        raise ValueError('Invalid frame dimensions')
    columns = math.ceil((max_x - min_x) / step)
    rows = math.ceil((max_z - min_z) / step)
    raw = source.read_bytes()
    if source.suffix == '.gz':
        raw = gzip.decompress(raw)
    heights = np.full((rows, columns), np.nan, dtype=np.float64)
    records = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    if not {'ix', 'iz', 'height'}.issubset(records.fieldnames or []):
        raise ValueError('CSV must contain ix, iz, height')
    for record in records:
        ix, iz, height = int(record['ix']), int(record['iz']), float(record['height'])
        if not (0 <= ix < columns and 0 <= iz < rows) or not math.isfinite(height):
            raise ValueError('Invalid cell index or height')
        if not np.isnan(heights[iz, ix]):
            raise ValueError(f'Duplicate cell: {ix}, {iz}')
        for axis, origin, index in (('x', min_x, ix), ('z', min_z, iz)):
            if axis in record and not math.isclose(
                    float(record[axis]), origin + (index + 0.5) * step,
                    rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'CSV {axis} disagrees with supplied grid')
        heights[iz, ix] = height
    if not np.isfinite(heights).all():
        raise ValueError('Incomplete grid; missing values must not become zero')
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'height.npy', heights, allow_pickle=False)
    low, high = float(heights.min()), float(heights.max())
    query_max_x = min_x + columns * step
    query_max_z = min_z + rows * step
    labels = {
        'en': ('Sampled terrain height', 'World Y, m',
               'Cell-centre samples; no smoothing. Colour does not indicate passability.'),
        'ru': ('Измеренная высота земли', 'Мировая высота Y, м',
               'Замеры в центрах клеток; без сглаживания. Цвет не означает проходимость.'),
    }
    for language, (title, colour_label, note) in labels.items():
        fig, ax = plt.subplots(figsize=(10, 9))
        plot = ax.imshow(heights, origin='lower', interpolation='nearest',
                         extent=(min_x, query_max_x, min_z, query_max_z),
                         cmap='viridis', vmin=low, vmax=high if high > low else low + 1)
        ax.set(xlim=(min_x, max_x), ylim=(min_z, max_z),
               xlabel='X, m', ylabel='Z, m',
               title=f'{title} | {step:g} m | {columns} × {rows}')
        ax.set_aspect('equal')
        fig.colorbar(plot, ax=ax, label=colour_label, shrink=0.85)
        fig.text(0.5, 0.035, note, ha='center', fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output / f'heightmap.{language}.png', dpi=160)
        plt.close(fig)
    metadata = {
        'source_csv': source.as_posix(),
        'source_csv_sha256': hashlib.sha256(raw).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'min_x': min_x, 'max_x': max_x, 'min_z': min_z, 'max_z': max_z,
        'query_max_x': query_max_x, 'query_max_z': query_max_z,
        'step_m': step, 'columns': columns, 'rows': rows, 'samples': int(heights.size),
        'array_order': 'height[iz, ix]; row 0 starts at min_z; column 0 at min_x',
        'sample_location': 'cell centre', 'units': 'metres',
        'min_height_m': low, 'max_height_m': high, 'height_range_m': high - low,
        'height_npy_sha256': hashlib.sha256((output / 'height.npy').read_bytes()).hexdigest(),
        'derived_offline': True, 'smoothing': False,
        'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
    }
    (output / 'heightmap.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bounds', nargs=4, type=float, required=True,
                        metavar=('MIN_X', 'MAX_X', 'MIN_Z', 'MAX_Z'))
    parser.add_argument('--step', required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(render(args.csv, args.output, args.bounds, args.step), indent=2))
