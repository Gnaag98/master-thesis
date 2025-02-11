import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    scripts_directory = Path(__file__).parent
    root_directory = scripts_directory.parent
    output_directory = root_directory / 'output'

    with open(output_directory / 'durations_repeated.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        rows = [row[:-1] for row in reader]
    durations_shared = rows[0]

    plt.plot(durations_shared, '*-')
    plt.legend('global', 'shared')
    plt.xlabel('i')
    plt.ylabel('Duration (µs)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
