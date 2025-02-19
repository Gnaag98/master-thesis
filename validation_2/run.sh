#!/bin/bash

# Parse command line arguments.
usage="Usage: ${BASH_SOURCE[0]} dim_x dim_y particles_per_cell version"
usage+=" distribution [positions_filepath]"
if [[ $# -lt 5 ]]; then
    echo ${usage}
    exit 1
fi
dim_x=$1
dim_y=$2
particles_per_cell=$3
version=$4
distribution=$5
positions_filepath=$6

# positions_filepath only used when distributing particles from a file.
if [[ (${distribution} -eq 2 && $# -lt 5) ]]; then
    echo "position_filepath required when distributing particles from a file."
    echo ${usage}
    exit 1
fi

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..

# Make sure the program is up to date.
${root}/build.sh
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Setup program parameters not specified by command line arguments.
dim_z=1
cell_size=64
# particles_per_cell not used when distributing particles from a file.
output_directory=${directory}/output
should_save=1

# Run the program.
program_output=$(${root}/build/master_thesis ${dim_x} ${dim_y} ${dim_z} \
    ${cell_size} ${particles_per_cell} ${version} ${output_directory} \
    ${distribution} ${should_save} ${positions_filepath})

# Get the total particle count.
particle_count=$(echo "${program_output}" | grep generated | tr -cd 0-9)

# Compute total charge in two different ways and compare.
python=${root}/.venv/bin/python
${python} ${directory}/compare.py ${dim_x} ${dim_y} ${version} ${particle_count}
