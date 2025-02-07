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
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    grid_size_x = args.dim_x + 2
    grid_size_y = args.dim_y + 2

    # maximum allowed error.
    tolerance = 0.001

    # Output directory relative to this file.
    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'

    # Get charge densities.
    with open(output_directory / 'charge_densities_global.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities_global = np.array([row[:-1] for row in reader][0])
    densities_global = densities_global[0:grid_size_x*grid_size_y]
    densities_global = np.reshape(densities_global, (-1, grid_size_x))
    with open(output_directory / 'charge_densities_shared.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        densities_shared = np.array([row[:-1] for row in reader][0])
    densities_shared = densities_shared[0:grid_size_x*grid_size_y]
    densities_shared = np.reshape(densities_shared, (-1, grid_size_x))
    absolute_errors = np.abs(densities_global - densities_shared)
    has_error = absolute_errors > tolerance

    print(f'Max absolute error: {absolute_errors.max()}')
    print(f'Number of cells with errors larger than the tolerance: {has_error.sum()}')

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes.
    ax = plt.axes()
    plt.xlim((-1, (grid_size_x-1)))
    plt.ylim((-1, (grid_size_y-1)))
    # Plot heatmap by drawing rectangles.
    for j in range(grid_size_y):
        for i in range(grid_size_x):
            ax.add_patch(Rectangle(((i-1), (j-1)), 1, 1, facecolor=f'{1-has_error[j,i]}'))
    # Plot borders to separate to distinguish the ghost cells.
    ax.add_patch(Rectangle((-1, -1), grid_size_x, grid_size_y, edgecolor='k', facecolor='none'))
    ax.add_patch(Rectangle((0, 0), grid_size_x-2, grid_size_y-2, edgecolor='r', facecolor='none'))
    # Show the figure.
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
