#!/bin/bash

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..
# Get file containing particle positions.
positions_filepath=${directory}/positions*x*x*.csv
if [ ! -f ${positions_filepath} ]; then
    echo count not find file matching ${positions_filepath}
    exit 1
fi


# Setup program parameters
# Filename must include the string {x}x{y}x{z} where {x}, {y}, {z} are integers.
dimensions=($( echo ${positions_filepath} | grep -oP '\d+x\dx+\d' | tr 'x' '\n' ))
dim_x=${dimensions[0]}
dim_y=${dimensions[1]}
dim_z=${dimensions[2]}
cell_size=64
# particles_per_cell not used when distributing particles from a file.
particles_per_cell=-1
version=0
output_directory=${directory}/output
distribution=2
should_save=1

# Make sure the program is up to date.
${root}/build.sh

# Run the global version.
${root}/build/master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} \
    ${particles_per_cell} ${version} ${output_directory} ${distribution} \
    ${should_save} ${positions_filepath}

# Plot charge densities.
${root}/.venv/bin/python ${directory}/plot.py ${dim_x} ${dim_y}
