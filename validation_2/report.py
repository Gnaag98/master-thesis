import argparse
import csv
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt


def get_data(filepath: Path):
    with open(filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row for row in reader]
        data: list[list[int|float]] = rows[1:]
    return data


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('version', type=str)
    parser.add_argument('distribution', type=str)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    directory: Path = args.directory
    version: str = args.version
    distribution: str = args.distribution

    filename_pattern = f'{version}_{distribution}_*.csv'
    nonrandom_filename_pattern = f'{version}_{distribution}_nonrandom_*.csv'
    filepaths_generator = directory.glob(filename_pattern)
    nonrandom_filepaths_generator = directory.glob(nonrandom_filename_pattern)
    nonrandom_filepaths = [p for p in nonrandom_filepaths_generator]
    # Exclude nonrandom from filepaths
    filepaths = [p for p in filepaths_generator if p not in nonrandom_filepaths]

    data_per_dim = {}
    for path in filepaths:
        match: re.Match = re.search(r'(\d+)x(\d+)[\D]+(\d+)', path.stem)
        dim_x = int(match.group(1))
        dim_y = int(match.group(2))
        run_index = int(match.group(3))

        data = get_data(path)
        iteration_count = len(data)
        expected: int = data[0][1]
        computed_values = np.array([float(row[2]) for row in data])
        relative_errors = np.mean(abs(1 - computed_values / expected))
        mean_relative_errors = np.mean(relative_errors)

        dim = f'{dim_x}x{dim_y}'
        if (dim not in data_per_dim):
            data_per_dim[dim] = []
        data_per_dim[dim].append({
            'dim_x': dim_x,
            'dim_y': dim_y,
            'index': run_index,
            'error': mean_relative_errors
        })

    # Sort the dimensions by run index so that the legend is ordered.
    sort_key = lambda data: (data[1][0]['index'])
    data_sorted = sorted(data_per_dim.items(), key=sort_key)

    for dim, data in data_sorted:
        a, b = sorted({ item['index']: item['error'] for item in data }.items())
        print(f'{a[0]}: {a[1]}')
        print(f'{b[0]}: {b[1]}')

    fig, ax = plt.subplots()
    bar_width = 0.8
    for dim, data in data_sorted:
        x = [item['index'] for item in data]
        y = [item['error']*100 for item in data]
        ax.bar(x, y, bar_width, label=dim)

    x_ticks = [item['index'] for data in data_per_dim.values() for item in data]
    x_tick_labels = [f'#{tick}' for tick in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Simulation run number')
    ax.set_ylabel('mean relative error [%]')
    ax.set_title(f'Total charge error')
    legend = ax.legend(title='Grid cells', framealpha=1)
    legend.get_title().set_fontsize('large')
    ax.grid(True, axis='y')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
