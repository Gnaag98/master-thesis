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

    run_data = []

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
            if dim != current_dim:
                current_dim = dim
                data[dim] = {}
                data[dim]['dim_x'] = row[0]
                data[dim]['dim_y'] = row[1]
                data[dim]['particles'] = row[3]
                data[dim]['expected'] = row[4]
                data[dim]['computed'] = []
            data[dim]['computed'].append(row[5])
            iteration = int(row[2])

        iteration_count = iteration + 1

        # Compute error.
        print(f'Relative error ({ppc} particles per cell)')
        for v in data.values():
            v['computed'] = np.array(v['computed'])
            v['error'] = np.abs(v['computed'] - v['expected'])
            v['error_max'] = np.max(v['error'])
            v['error_mean'] = np.mean(v['error'])
            v['error_relative'] = v['error_mean'] / v['expected']
            print(f"  {v['dim_x']}x{v['dim_y']}: {v['error_mean'] / v['expected']}")
        run_data.append(data)
        # Plot
        ticks = np.arange(len(data))
        bar_width = 0.25
        bar_labels = [k for k in data]
        bar_expected = [v['expected'] for v in data.values()]
        bar_max = np.divide([v['error_max'] for v in data.values()], bar_expected)
        bar_mean = np.divide([v['error_mean'] for v in data.values()], bar_expected)

        def log2(x):
            f = lambda x : -np.log2(np.abs(x)) if x != 0 else 0
            try:
                return [f(v) for v in x]
            except TypeError:
                return f(x)

        fig, ax = plt.subplots()
        ax.bar(ticks - bar_width / 2, log2(bar_max), bar_width, label='Max')
        ax.bar(ticks + bar_width / 2, log2(bar_mean), bar_width, label=f'Mean')

        ax.set_xlabel('Cell configuration')
        ax.set_xticks(ticks)
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel('-log2(relative error)')
        ax.set_title(f'Total charge error, {iteration_count} iterations, {ppc} particles per cell')
        ax.grid(True, axis='y')
        ax.legend()
        fig.tight_layout()

    is_pow2 = lambda n : (n & (n - 1) == 0) and n != 0

    # Combined log plot.
    ticks = set()
    fig, ax = plt.subplots()
    for ppc, data in zip(particles_per_cell, run_data):
        # Only show square grid configurations
        are_square = np.all([v['dim_x'] == v['dim_y'] for v in data.values()])
        if not are_square:
            continue
        
        #x = [np.log2(v['dim_x']) for v in data.values()]
        x = [np.log2(v['dim_x']**2) for v in data.values()]
        for tick in x:
            ticks.add(int(tick))
        y = [-np.log2(v['error_relative']) for v in data.values()]
        style = 'o-' if is_pow2(ppc) else '*-'

        ax.plot(x, y, style, label=f'{ppc} particles per cell')
    
    #ticks = sorted(ticks)
    #labels = [f'{2**tick}x{2**tick}' for tick in ticks]
    #ax.set_xticks(ticks)
    #ax.set_xticklabels(labels)
    #ax.set_xlabel('Cell configuration')
    ax.set_xlabel('log2(number of cells)')
    ax.set_ylabel('-log2(mean relative error)')
    ax.set_title(f'Total charge error, {iteration_count} iterations')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Percentage plot.
    ticks = set()
    fig, ax = plt.subplots()
    for ppc, data in zip(particles_per_cell, run_data):
        # Only show square grid configurations
        are_square = np.all([v['dim_x'] == v['dim_y'] for v in data.values()])
        if not are_square:
            continue
        
        x = [v['dim_x'] for v in data.values()]
        for tick in x:
            ticks.add(int(tick))
        y = [v['error_relative']*100 for v in data.values()]
        style = 'o-' if is_pow2(ppc) else '*-'

        ax.plot(x, y, style, label=f'{ppc} particles per cell')
    
    ticks = sorted(ticks)
    labels = [f'{tick}' for tick in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Cell-grid side length')
    ax.set_ylabel('mean relative error (%)')
    ax.set_title(f'Total charge error, {iteration_count} iterations')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    


if __name__ == '__main__':
    main()
