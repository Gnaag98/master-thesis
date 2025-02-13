import argparse
import csv
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

def plot_densities(densities_filepath: Path, grid_size_x: int, grid_size_y: int,
                   x: list, y: list, figures_directory: Path):
    # Get charge densities.
    with open(densities_filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities = np.array([row[:-1] for row in reader][0])
    densities = densities[0:grid_size_x*grid_size_y]
    densities = np.reshape(densities, (-1, grid_size_x))

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes.
    fig, ax = plt.subplots()
    ax.set_xlim((-1, (grid_size_x-1)))
    ax.set_ylim((-1, (grid_size_y-1)))
    ax.axis('equal')
    ax.set_title('Validation 1')
    # Plot heatmap by drawing rectangles.
    for j in range(grid_size_y):
        for i in range(grid_size_x):
            density = densities[j,i]
            density_normalized = density / densities.max()
            ax.add_patch(Rectangle(((i-1), (j-1)), 1, 1, facecolor=f'{1 - density_normalized}'))
            if density_normalized > 0.5:
                text_color='w'
            else:
                text_color='k'
            if density_normalized != 0:
                ax.text(i-0.5, j-0.5, f'{density_normalized:.3f}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8, color=text_color)
    # Plot borders to separate to distinguish the ghost cells.
    ax.add_patch(Rectangle((-1, -1), grid_size_x, grid_size_y, edgecolor='k', facecolor='none'))
    ax.add_patch(Rectangle((0, 0), grid_size_x-2, grid_size_y-2, edgecolor='r', facecolor='none'))
    # Plot particle positions.
    ax.scatter(x, y, s=[8*4 for _ in range(len(x))])
    # Save the figure.
    fig.savefig(figures_directory / f'charge_densities_global.png')


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    cell_size = 64
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2

    # Directories relative to this file.
    directory = Path(__file__).parent
    positions_filepath = directory / 'output' / 'positions.csv'
    densities_filepath = directory / 'output' / 'charge_densities_global.csv'
    figures_directory = directory / 'figures'

    # Get particle positions.
    with open(positions_filepath) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        positions = [row[:-1] for row in reader]
    x = [x / cell_size for x in positions[0]]
    y = [y / cell_size for y in positions[1]]
    
    plot_densities(densities_filepath, grid_size_x, grid_size_y, x, y,
                   figures_directory)
    plt.show()


if __name__ == '__main__':
    main()
