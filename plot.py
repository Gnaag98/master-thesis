import csv
import matplotlib.pyplot as plt


def main():
    with open('output/positions.csv') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        positions = [row[:-1] for row in reader]
    x = positions[0]
    y = positions[1]
    z = positions[2]

    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    main()
