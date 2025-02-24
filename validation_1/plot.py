import argparse
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


def plot_densities(densities_filepath: Path, grid_size_x: int, grid_size_y: int,
                   x: list, y: list, figure_filename: Path):
    # Get charge densities.
    densities = np.load(densities_filepath)
    densities = densities[0:grid_size_x*grid_size_y]
    densities: np.ndarray = np.reshape(densities, (-1, grid_size_x))

    # Initialize size of figure before drawing rectangles since add_patch
    # doesn't resize the axes, including padding outside grid.
    fig, ax = plt.subplots()
    ax.set_xlim((-1.5, (grid_size_x-0.5)))
    ax.set_ylim((-1.5, (grid_size_y-0.5)))
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
    fig.savefig(figure_filename)


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    parser.add_argument('version', type=str)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    cell_size = 64
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2
    version: str = args.version

    # Directories relative to this file.
    directory = Path(__file__).parent
    positions_filepath = directory / 'output' / 'positions.npz'
    densities_filepath = directory / 'output' / f'charge_densities_{version}.npy'
    figures_directory = directory / 'output'
    figures_directory.mkdir(exist_ok=True)
    figure_filename = figures_directory / f'charge_densities_{version}.png'

    # Get particle positions.
    positions = np.load(positions_filepath)
    x: np.ndarray = positions['pos_x']
    y: np.ndarray = positions['pos_y']
    x /= cell_size
    y /= cell_size

    plot_densities(densities_filepath, grid_size_x, grid_size_y, x, y,
                   figure_filename)
    plt.show()


if __name__ == '__main__':
    main()
