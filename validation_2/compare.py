import argparse
from pathlib import Path

import numpy as np


def get_densities(densities_filepath: Path, grid_size_x: int, grid_size_y: int):
    densities: np.ndarray = np.load(densities_filepath)
    densities = densities[0:grid_size_x*grid_size_y]
    densities: np.ndarray = np.reshape(densities, (-1, grid_size_x))
    return densities


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('dim_x', type=int)
    parser.add_argument('dim_y', type=int)
    parser.add_argument('version', type=int)
    parser.add_argument('particle_count', type=int)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2
    if args.version == 0:
        version = 'global'
    else:
        version = 'shared'
    particle_count: int = args.particle_count

    particle_charge = 1.0


    # Directories relative to this file.
    directory = Path(__file__).parent
    densities_filepath = directory / 'output' / f'charge_densities_{version}.npy'

    densities = get_densities(densities_filepath, grid_size_x, grid_size_y)
    
    total_charge_product = particle_count * particle_charge
    total_charge_sum = densities.sum()
    error_absolute = abs(total_charge_sum - total_charge_product)
    error_relative = error_absolute / total_charge_sum
    print(f'particle_count: {particle_count}')
    print(f'total_charge(product): {total_charge_product}')
    print(f'total_charge(sum): {total_charge_sum}')
    print(f'error_absolute: {error_absolute}')
    print(f'error_relative: {error_relative}')


if __name__ == '__main__':
    main()
