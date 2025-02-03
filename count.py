import csv
from math import floor
import matplotlib.pyplot as plt


def main():
    with open('output/positions.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        positions = [row[:-1] for row in reader]
    pos_x = positions[0]
    pos_y = positions[1]
    pos_z = positions[2]

    cell_size = 64
    grid_size_x = 2
    grid_size_y = 4
    grid_size_z = 8
    cells_python = [0 for i in range(grid_size_x*grid_size_y*grid_size_z)]
    for cell_index in range(len(cells_python)):
        i = cell_index % grid_size_x
        j = floor(cell_index / grid_size_x) % grid_size_y
        k = floor(cell_index / grid_size_x / grid_size_y) % grid_size_z

        x_min = cell_size * i
        x_max = cell_size * (i + 1)
        y_min = cell_size * j
        y_max = cell_size * (j + 1)
        z_min = cell_size * k
        z_max = cell_size * (k + 1)

        for particle_index in range(len(pos_x)):
            x = pos_x[particle_index]
            y = pos_y[particle_index]
            z = pos_z[particle_index]
            if (x_min < x < x_max and y_min < y < y_max and z_min < z < z_max):
                cells_python[cell_index] += 1


    with open('output/particle_counts.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        cells_cuda = [row[:-1] for row in reader][0]

    absolute_error = sum([a - b for a, b in zip(cells_cuda, cells_python)])

    print(f'Absoulte error: {absolute_error}')

if __name__ == '__main__':
    main()
