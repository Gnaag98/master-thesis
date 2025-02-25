import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def get_data(filepath: Path):
    with open(filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row for row in reader]
        data: list[list[int]] = rows[1:]
    return data


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('version', type=str)
    parser.add_argument('distribution', type=str)
    parser.add_argument('particles_per_cell', type=int, action="extend", nargs="+")
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    directory: Path = args.directory
    version: str = args.version
    distribution: str = args.distribution
    particles_per_cell: list[int] = args.particles_per_cell

    for ppc in particles_per_cell:
        filename = f'{version}_{distribution}_{ppc}ppc.csv'
        filepath = directory / filename
        file_data = get_data(filepath)
        data = {}
        current_dim = ''
        iteration = 0
        # Parse rows into a dictionary.
        for row in file_data:
            dim = f'{int(row[0])}x{int(row[1])}'
            if (dim != current_dim):
                current_dim = dim
                data[dim] = {}
                data[dim]['dim_x'] = row[0]
                data[dim]['dim_y'] = row[1]
                data[dim]['particles'] = row[3]
                data[dim]['product'] = row[4]
                data[dim]['sum'] = []
            data[dim]['sum'].append(row[5])
            iteration = int(row[2])

        iteration_count = iteration + 1

        # Compute error.
        print(f'Relative error ({ppc} particles per cell)')
        for v in data.values():
            v['sum'] = np.array(v['sum'])
            v['error'] = v['product'] - v['sum']
            v['error_min'] = np.min(v['error'])
            v['error_max'] = np.max(v['error'])
            v['error_mean'] = np.mean(v['error'])
            print(f"  {v['dim_x']}x{v['dim_y']}: {v['error_mean'] / v['product']}")

        # Plot
        ticks = np.arange(len(data))
        bar_width = 0.25
        bar_product = [v['product'] for v in data.values()]
        bar_min = np.divide([v['error_min'] for v in data.values()], bar_product)
        bar_max = np.divide([v['error_max'] for v in data.values()], bar_product)
        bar_mean = np.divide([v['error_mean'] for v in data.values()], bar_product)
        fig, ax = plt.subplots()
        ax.bar(ticks - bar_width, bar_min, bar_width, label='Min')
        ax.bar(ticks, bar_max, bar_width, label='Max')
        ax.bar(ticks + bar_width, bar_mean, bar_width, label=f'Mean')
        ax.set_xlabel('Cell configuration')
        ax.set_xticks(ticks)
        ax.set_xticklabels([k for k in data])
        ax.set_ylabel('(product - sum) / product')
        ax.set_title(f'Total charge error, {iteration_count} iterations, {ppc} particles per cell')
        ax.grid(True, axis='y')
        ax.legend()
        fig.tight_layout()

    plt.show()
    


if __name__ == '__main__':
    main()
