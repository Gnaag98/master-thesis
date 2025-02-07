import argparse
import csv
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    parser.add_argument('filename')
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    cell_size = 64
    grid_size_x = args.dim_x + 2
    grid_size_y = args.dim_y + 2

    densities_filename = args.filename

    # Output directory relative to this file.
    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'

    # Get particle positions.
    with open(output_directory / 'positions.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        positions = [row[:-1] for row in reader]
    x = [x / cell_size for x in positions[0]]
    y = [y / cell_size for y in positions[1]]
    # Get charge densities.
    with open(output_directory / densities_filename) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities = np.array([row[:-1] for row in reader][0])
    densities = densities[0:grid_size_x*grid_size_y]
    densities = np.reshape(densities, (-1, grid_size_x))

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes.
    ax = plt.axes()
    plt.xlim((-1, (grid_size_x-1)))
    plt.ylim((-1, (grid_size_y-1)))
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
    #ax.scatter(x, y, s=[8*4 for _ in range(len(x))])
    # Show the figure.
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
