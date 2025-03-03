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
    parser.add_argument('--title', type=str)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    directory: Path = args.directory
    version: str = args.version
    distribution: str = args.distribution
    title_extra: str = args.title

    filename_pattern = f'{version}_{distribution}_*.csv'
    nonrandom_filename_pattern = f'{version}_{distribution}_nonrandom_*.csv'
    filepaths_generator = directory.glob(filename_pattern)
    nonrandom_filepaths_generator = directory.glob(nonrandom_filename_pattern)
    nonrandom_filepaths = [p for p in nonrandom_filepaths_generator]
    # Exclude nonrandom from filepaths
    filepaths = [p for p in filepaths_generator if p not in nonrandom_filepaths]

    # Data from runs with random seeds.
    data_per_dim = {}
    for path in filepaths:
        match: re.Match = re.search(r'(\d+)x(\d+)[\D]+(\d+)', path.stem)
        dim_x = int(match.group(1))
        dim_y = int(match.group(2))
        run_index = int(match.group(3))

        data = get_data(path)
        expected = int(data[0][1])
        computed_values = np.array([float(row[2]) for row in data])
        mean_relative_error = np.mean(abs(1 - computed_values / expected))
        relative_error_of_mean = abs(1 - np.mean(computed_values) / expected)

        dim = f'{dim_x}x{dim_y}'
        if (dim not in data_per_dim):
            data_per_dim[dim] = []
        data_per_dim[dim].append({
            'dim_x': dim_x,
            'dim_y': dim_y,
            'index': run_index,
            'mean_relative_error': mean_relative_error,
            'relative_error_of_mean': relative_error_of_mean,
        })
    
    # Data from runs with the same seed.
    nonrandom_data_per_dim = {}
    for path in nonrandom_filepaths:
        match: re.Match = re.search(r'(\d+)x(\d+)[\D]+(\d+)', path.stem)
        dim_x = int(match.group(1))
        dim_y = int(match.group(2))
        run_index = int(match.group(3))

        data = get_data(path)
        expected = int(data[0][1])
        computed_values = np.array([float(row[2]) for row in data])
        unique_values_count = len(set(computed_values))

        dim = f'{dim_x}x{dim_y}'
        if (dim not in nonrandom_data_per_dim):
            nonrandom_data_per_dim[dim] = []
        nonrandom_data_per_dim[dim].append({
            'dim_x': dim_x,
            'dim_y': dim_y,
            'index': run_index,
            'unique_values_count': unique_values_count,
        })

    # Sort the dimensions by run index so that the legend is ordered.
    sort_key = lambda data: (data[1][0]['index'])
    data_sorted = sorted(data_per_dim.items(), key=sort_key)
    nonrandom_data_sorted = sorted(nonrandom_data_per_dim.items(), key=sort_key)

    # Mean relative error.
    fig, ax = plt.subplots()
    bar_width = 0.8
    for dim, data in data_sorted:
        x = [item['index'] for item in data]
        y = [item['mean_relative_error']*100 for item in data]
        ax.bar(x, y, bar_width, label=dim)

    x_ticks = [item['index'] for data in data_per_dim.values() for item in data]
    x_tick_labels = [f'#{tick}' for tick in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Simulation run number')
    ax.set_ylabel('Mean relative error [%]')
    title = 'Total charge error' + (f' ({title_extra})' if title_extra else '')
    ax.set_title(title)
    legend = ax.legend(title='Grid cells', loc='upper left')
    legend.get_title().set_fontsize('large')
    ax.grid(True, axis='y')
    fig.tight_layout()

    # Relative error of mean.
    fig, ax = plt.subplots()
    bar_width = 0.8
    for dim, data in data_sorted:
        x = [item['index'] for item in data]
        y = [item['relative_error_of_mean']*100 for item in data]
        ax.bar(x, y, bar_width, label=dim)

    x_ticks = [item['index'] for data in data_per_dim.values() for item in data]
    x_tick_labels = [f'#{tick}' for tick in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Simulation run number')
    ax.set_ylabel('Relative error of mean [%]')
    title = 'Total charge error' + (f' ({title_extra})' if title_extra else '')
    ax.set_title(title)
    legend = ax.legend(title='Grid cells', loc='upper left')
    legend.get_title().set_fontsize('large')
    ax.grid(True, axis='y')
    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
