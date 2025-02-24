#!/bin/bash

# Assumes positional arguments before optional arguments.
dim_x=$1
dim_y=$2

# Get directory of script independent of working directory.
directory=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root=${directory}/..

version=global
output_directory=${directory}/output

# Parse arguments to remove those overwritten by this script.
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            # Using hardcoded directory so the user should not overwrite it.
            shift
            shift
            ;;
        -v)
            # Replace default with user defined value.
            version=$2
            shift
            shift
            ;;
        *)
            args+=($1)
            shift
            ;;
    esac
done

# Make sure the program is up to date.
${root}/build.sh
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Run the program.
program_output=$(
    ${root}/build/master_thesis ${args[*]} -o ${output_directory} \
    -v ${version} \
)
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Get the total particle count.
particle_count=$(echo "${program_output}" | grep generated | tr -cd 0-9)

# Compute total charge in two different ways and compare.
python=${root}/.venv/bin/python
${python} ${directory}/compare.py ${dim_x} ${dim_y} ${version} ${particle_count}
