#!/bin/bash

# Parse command line arguments.
if [[ $# -ne 1 ]]; then
    echo "usage: ${BASH_SOURCE[0]} version"
    exit 1
fi
version=$1

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..

# Filename must include the string {x}x{y}x{z} where {x}, {y}, {z} are integers.
positions_filepath=${directory}/positions*x*x*.csv
if [ ! -f ${positions_filepath} ]; then
    echo count not find file matching ${positions_filepath}
    exit 1
fi
dimensions=($( \
    echo ${positions_filepath} \
    | grep -oP '\d+x\dx+\d' \
    | tr 'x' '\n' \
))

dim_x=${dimensions[0]}
dim_y=${dimensions[1]}
dim_z=${dimensions[2]}
output_directory=${directory}/output

# Make sure the program is up to date.
${root}/build.sh
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Run the program.
${root}/build/master_thesis ${dim_x} ${dim_y} ${dim_z} \
    -v ${version} -o ${output_directory} -d ${positions_filepath}
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Plot charge densities and compare with expected values.
python=${root}/.venv/bin/python
${python} ${directory}/plot.py ${dim_x} ${dim_y} ${version}
${python} ${directory}/compare.py ${dim_x} ${dim_y} ${version}
