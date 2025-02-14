import argparse
import csv
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


def heatmap(grid_data: np.ndarray, figures_directory: Path):
    grid_size_y, grid_size_x = grid_data.shape
    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes, including padding outside grid.
    fig, ax = plt.subplots()
    ax.set_xlim((-1.5, (grid_size_x-0.5)))
    ax.set_ylim((-1.5, (grid_size_y-0.5)))
    ax.axis('equal')
    ax.set_title('Validation 1 - normalized errors')
    # Plot heatmap by drawing rectangles.
    for j in range(grid_size_y):
        for i in range(grid_size_x):
            data = grid_data[j,i]
            if (grid_data.max() != 0):
                data = data / grid_data.max()
            ax.add_patch(Rectangle(((i-1), (j-1)), 1, 1, facecolor=f'{1 - abs(data)}'))
            if data > 0.5:
                text_color='w'
            else:
                text_color='k'
            if data != 0:
                ax.text(i-0.5, j-0.5, f'{data:.3f}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8, color=text_color)
    # Plot borders to separate to distinguish the ghost cells.
    ax.add_patch(Rectangle((-1, -1), grid_size_x, grid_size_y, edgecolor='k', facecolor='none'))
    ax.add_patch(Rectangle((0, 0), grid_size_x-2, grid_size_y-2, edgecolor='r', facecolor='none'))
    # Save the figure.
    fig.savefig(figures_directory / f'charge_densities_global_error.png')
    plt.show()


def get_densities_npy(densities_filepath: Path, grid_size_x: int,
                      grid_size_y: int):
    densities = np.load(densities_filepath)
    densities = densities[0:grid_size_x*grid_size_y]
    densities: np.ndarray = np.reshape(densities, (-1, grid_size_x))
    return densities


def get_densities_csv(densities_filepath: Path, grid_size_x: int,
                      grid_size_y: int):
    with open(densities_filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities = np.array([row[:-1] for row in reader][0])
    densities = densities[0:grid_size_x*grid_size_y]
    densities: np.ndarray = np.reshape(densities, (-1, grid_size_x))
    return densities


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2

    # Directories relative to this file.
    directory = Path(__file__).parent
    key_directory = directory / 'key'
    run_directory = directory / 'output'
    figures_directory = directory / 'output'
    figures_directory.mkdir(exist_ok=True)
    densities_filename_stem = 'charge_densities_global'

    run_densities = get_densities_npy(
        run_directory / f'{densities_filename_stem}.npy', grid_size_x,
        grid_size_y)
    key_densities = get_densities_csv(
        key_directory / f'{densities_filename_stem}.csv', grid_size_x,
        grid_size_y)
    
    error = np.abs(run_densities - key_densities)
    min_error = error.min()
    max_error = error.max()
    mean_error = error.mean()
    print(f'min_error: {min_error}')
    print(f'max_error: {max_error}')
    print(f'avg_error: {mean_error}')
    if min_error or max_error:
        heatmap(error, figures_directory)


if __name__ == '__main__':
    main()
