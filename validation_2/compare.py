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
    parser.add_argument('version', type=str)
    parser.add_argument('particle_count', type=int)
    args = parser.parse_args()
    # Grid parameters with ghost cells included.
    grid_size_x: int = args.dim_x + 2
    grid_size_y: int = args.dim_y + 2
    version = args.version
    particle_count: int = args.particle_count

    # Directories relative to this file.
    directory = Path(__file__).parent
    densities_filepath = directory / 'output' / f'charge_densities_{version}.npy'

    densities = get_densities(densities_filepath, grid_size_x, grid_size_y)
    
    total_charge_expected = particle_count
    total_charge_computed = densities.sum()
    error = total_charge_computed - total_charge_expected
    error_absolute = abs(error)
    error_relative = error_absolute / total_charge_computed
    print(f'particle_count: {particle_count}')
    print(f'total_charge(expected): {total_charge_expected}')
    print(f'total_charge(computed): {total_charge_computed}')
    print(f'error: {error}')
    print(f'error_absolute: {error_absolute}')
    print(f'error_relative: {error_relative}')


if __name__ == '__main__':
    main()
