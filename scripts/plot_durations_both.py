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
    filename = f'durations_both_{particles_per_cell}ppc.csv'

    with open(output_directory / filename) as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row for row in reader]
    dimensions = rows[0]
    particle_count = np.pow(dimensions, 2) * particles_per_cell
    durations_global = np.array(rows[1])
    durations_shared = np.array(rows[2])

    difference = durations_global - durations_shared

    _, ax = plt.subplots()
    ax.plot(particle_count, durations_global / 1000, 'o-', label='Global')
    ax.plot(particle_count, durations_shared / 1000, '*-', label='Shared')
    ax.set_xlabel('Number of particles')
    ax.set_ylabel('Duration (ms)')
    ax.set_title(f'{particles_per_cell} particles per cell')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(2e7, 6e7)
    ax.set_ylim(0, 140)
    plt.show()


if __name__ == '__main__':
    main()
