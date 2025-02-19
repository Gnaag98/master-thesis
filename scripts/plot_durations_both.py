import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'

    with open(output_directory / 'durations_both.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row for row in reader]
    dimensions = rows[0]
    durations_global = np.array(rows[1])
    durations_shared = np.array(rows[2])

    difference = durations_global - durations_shared
    performance_gain = np.divide(difference, durations_global)

    _, ax = plt.subplots()
    ax.plot(np.log2(dimensions), durations_global, 'o-', label='Global')
    ax.plot(np.log2(dimensions), durations_shared, '*-', label='Shared')
    ax.set_xlabel('log2(grid dimension) (square grid)')
    ax.set_ylabel('Duration (µs)')
    ax.legend()
    ax.grid(True)

    _, ax = plt.subplots()
    ax.plot(np.log2(dimensions), performance_gain*100, '*-')
    ax.set_xticks(np.log2(dimensions))
    ax.set_xlabel('log2(grid dimension) (square grid)')
    ax.set_ylabel('Performance gain (%)')
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
