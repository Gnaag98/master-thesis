import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('particles_per_cell', type=int)
    args = parser.parse_args()
    particles_per_cell: int = args.particles_per_cell

    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'
    filename = f'durations_shared_{particles_per_cell}ppc.csv'

    with open(output_directory / filename) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row for row in reader]
    dimensions = rows[0]
    particle_count = np.pow(dimensions, 2) * particles_per_cell
    durations_associate = np.array(rows[1])
    durations_sort = np.array(rows[2])
    durations_contextualize = np.array(rows[3])
    durations_density = np.array(rows[4])

    _, ax = plt.subplots()
    ax.plot(particle_count, durations_associate / 1000, 'o-', label='Associate')
    ax.plot(particle_count, durations_sort / 1000, 'o-', label='Sort')
    ax.plot(particle_count, durations_contextualize / 1000, 'o-', label='Contextualize')
    ax.plot(particle_count, durations_density / 1000, 'o-', label='Density')
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Duration (ms)')
    ax.set_title(f'Shared - {particles_per_cell} particles per cell')
    ax.legend()
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
