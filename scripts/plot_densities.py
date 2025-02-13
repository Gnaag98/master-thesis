import argparse
import csv
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

# Output directory relative to this file.
scripts_directory = Path(__file__).parent
root_directory = scripts_directory.parent


def plot_densities(file_directory: Path, version: str, grid_size_x: int,
                   grid_size_y: int, should_show_positions: bool, x: list,
                   y: list, save_directory: Path|None):
    # Get charge densities.
    densities_filename = f'charge_densities_{version}.csv'
    with open(file_directory / densities_filename) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities = np.array([row[:-1] for row in reader][0])
    densities = densities[0:grid_size_x*grid_size_y]
    densities = np.reshape(densities, (-1, grid_size_x))

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes.
    fig, ax = plt.subplots()
    ax.set_xlim((-1, (grid_size_x-1)))
    ax.set_ylim((-1, (grid_size_y-1)))
    # Plot heatmap by drawing rectangles.
    for j in range(grid_size_y):
        for i in range(grid_size_x):
            density = densities[j,i]
            density_normalized = density / densities.max()
            ax.add_patch(Rectangle(((i-1), (j-1)), 1, 1, facecolor=f'{1 - density_normalized}'))
    # Plot borders to separate to distinguish the ghost cells.
    ax.add_patch(Rectangle((-1, -1), grid_size_x, grid_size_y, edgecolor='k', facecolor='none'))
    ax.add_patch(Rectangle((0, 0), grid_size_x-2, grid_size_y-2, edgecolor='r', facecolor='none'))
    # Plot particle positions.
    if (should_show_positions):
        ax.scatter(x, y, s=[8*4 for _ in range(len(x))])
    # Show the figure.
    ax.set_title(version.capitalize())
    ax.axis('equal')
    if (save_directory is not None):
        fig.savefig(save_directory / f'charge_densities_{version}')


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    parser.add_argument('--directory', default='output')
    parser.add_argument('--positions', action='store_true')
    parser.add_argument('--save')
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    cell_size = 64
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2
    directory_argument = Path(args.directory)
    should_show_positions: bool = args.positions
    if (args.save):
        save_directory = Path(args.save)
    else:
        save_directory = None

    if (directory_argument.is_absolute()):
        file_directory = directory_argument
    else:
        file_directory = root_directory / directory_argument

    # Get particle positions.
    with open(file_directory / 'positions.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        positions = [row[:-1] for row in reader]
    x = [x / cell_size for x in positions[0]]
    y = [y / cell_size for y in positions[1]]
    
    plot_densities(file_directory, 'global', grid_size_x, grid_size_y,
                   should_show_positions, x, y, save_directory)
    plot_densities(file_directory, 'shared', grid_size_x, grid_size_y,
                   should_show_positions, x, y, save_directory)
    plt.show()


if __name__ == '__main__':
    main()
