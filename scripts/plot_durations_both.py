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
        rows = [row[:-1] for row in reader]
    dimensions = rows[0]
    durations_global = np.array(rows[1])
    durations_shared = np.array(rows[2])

    difference = durations_global - durations_shared
    performance_gain = np.divide(difference, durations_global)

    plt.plot(np.log2(dimensions), performance_gain*100, '*-')
    plt.legend('global', 'shared')
    plt.xticks(np.log2(dimensions))
    plt.xlabel('log2(grid dimension) (square grid)')
    plt.ylabel('Performance gain (%)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
