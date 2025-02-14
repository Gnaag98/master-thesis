#!/bin/bash

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..

# Setup program parameters
dim_x=$1
dim_y=$2
dim_z=1
cell_size=64
# particles_per_cell not used when distributing particles from a file.
particles_per_cell=$3
version=0
output_directory=${directory}/output
distribution=$4
should_save=1
# positions_filepath only used when distributing particles from a file.
positions_filepath=$5

# Make sure enough arguments are supplied.
usage="Usage: ${BASH_SOURCE[0]} dim_x dim_y particles_per_cell distribution"
usage+=" [positions_filepath]"
if [[ $# -lt 4 ]]; then
    echo ${usage}
    exit 1
fi
if [[ (${distribution} -eq 2 && $# -lt 5) ]]; then
    echo "position_filepath required when distributing particles from a file."
    echo ${usage}
    exit 1
fi

# Make sure the program is up to date.
${root}/build.sh
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Run the global version.
program_output=$(${root}/build/master_thesis ${dim_x} ${dim_y} ${dim_z} \
    ${cell_size} ${particles_per_cell} ${version} ${output_directory} \
    ${distribution} ${should_save} ${positions_filepath})

# Get the total particle count.
particle_count=$(echo "${program_output}" | grep generated | tr -cd 0-9)

# Compute total charge in two different ways and compare.
python=${root}/.venv/bin/python
${python} ${directory}/compare.py ${dim_x} ${dim_y} ${particle_count}
