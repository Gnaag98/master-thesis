import argparse
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

# Output directory relative to this file.
scripts_directory = Path(__file__).parent
root_directory = scripts_directory.parent


def plot(grid_size_x: int, grid_size_y: int, x: list, y: list):
    # Get charge densities.

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes, including padding outside grid.
    fig, ax = plt.subplots()
    ax.set_xlim((-1.5, (grid_size_x-0.5)))
    ax.set_ylim((-1.5, (grid_size_y-0.5)))
    ax.axis('equal')
    ax.set_title('Particle-cell association')

    # Plot grid lines manually since ax.grid(True) partly covers the rectangles.
    for i in range(1, grid_size_x - 2):
        line_x = [i, i]
        line_y = [-1, grid_size_y - 1]
        ax.plot(line_x, line_y, color='0.75', linewidth=0.75)
    for i in range(1, grid_size_y - 2):
        line_x = [-1, grid_size_x - 1]
        line_y = [i, i]
        ax.plot(line_x, line_y, color='0.75', linewidth=0.75)
    # Overlapping lines misbehave. Draw discontinuous line where needed.
    ax.plot([0, 0], [-1, 0], color='0.75', linewidth=0.75)
    ax.plot([0, 0], [grid_size_y - 2, grid_size_y - 1], color='0.75', linewidth=0.75)
    ax.plot([grid_size_x - 2, grid_size_x - 2], [-1, 0], color='0.75', linewidth=0.75)
    ax.plot([grid_size_x - 2, grid_size_x - 2], [grid_size_y - 2, grid_size_y - 1], color='0.75', linewidth=0.75)
    ax.plot([0, 0], [grid_size_y - 2, grid_size_y - 1], color='0.75', linewidth=0.75)
    ax.plot([-1, 0], [0, 0], color='0.75', linewidth=0.75)
    ax.plot([grid_size_x - 2, grid_size_x - 1], [0, 0], color='0.75', linewidth=0.75)
    ax.plot([-1, 0], [grid_size_y - 2, grid_size_y - 2], color='0.75', linewidth=0.75)
    ax.plot([grid_size_x - 2, grid_size_x - 1], [grid_size_y - 2, grid_size_y - 2], color='0.75', linewidth=0.75)
    
    # Plot borders to separate to distinguish the ghost cells.
    ax.add_patch(Rectangle((-1, -1), grid_size_x, grid_size_y, edgecolor='k', facecolor='none'))
    ax.add_patch(Rectangle((0, 0), grid_size_x-2, grid_size_y-2, edgecolor='r', facecolor='none'))
    # Draw between cell centers.
    for j in range(grid_size_y - 1):
        for i in range(grid_size_x - 1):
            ax.plot([i - 0.5, i - 0.5], [j - 0.5, j + 0.5], color='0.9', linestyle='dashed')
            ax.plot([i - 0.5, i + 0.5], [j - 0.5, j - 0.5], color='0.9', linestyle='dashed')
    ax.plot([ grid_size_x - 1.5, grid_size_x - 1.5], [-0.5, grid_size_y - 1.5], color='0.9', linestyle='dashed')
    ax.plot([-0.5, grid_size_x - 1.5], [ grid_size_y - 1.5, grid_size_y - 1.5], color='0.9', linestyle='dashed')
    # Plot cell indices at cell centers
    for j in range(grid_size_y):
        for i in range(grid_size_x):
            text = f'{i + j * grid_size_x}'
            ax.text(i - 0.5, j - 0.5, text, fontsize=9,
                    horizontalalignment='center', verticalalignment='center')
    # Plot particle index.
    for i in range(len(x)):
        ax.plot(x[i], y[i], 'ko', markersize=10)
        ax.text(x[i], y[i], f'{i}', color='w', fontsize=7,
                horizontalalignment='center', verticalalignment='center')


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    parser.add_argument('--directory', default='output')
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    cell_size = 64
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2
    directory_argument = Path(args.directory)

    if (directory_argument.is_absolute()):
        file_directory = directory_argument
    else:
        file_directory = root_directory / directory_argument

    # Get particle positions.
    positions = np.load(file_directory / 'positions.npz')
    x: np.ndarray = positions['pos_x']
    y: np.ndarray = positions['pos_y']
    x /= cell_size
    y /= cell_size

    # Get sorted indices
    particle_indices = np.load(file_directory / 'particle_indices.npy')
    cell_indices = np.load(file_directory / 'associated_cells.npy')
    particle_indices_sorted = np.load(file_directory / 'particle_indices_sorted.npy')
    cell_indices_sorted = np.load(file_directory / 'associated_cells_sorted.npy')
    print('Unsorted:')
    print(np.matrix([particle_indices, cell_indices]))
    print('Sorted:')
    print(np.matrix([particle_indices_sorted, cell_indices_sorted]))

    plot(grid_size_x, grid_size_y, x, y)
    plt.show()


if __name__ == '__main__':
    main()
