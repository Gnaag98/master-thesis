from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'

    x, y, z = np.load(output_directory / 'positions.npy')

    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    main()
