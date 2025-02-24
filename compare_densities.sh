#!/bin/bash

# Assumes positional arguments before optional arguments.
dim_x=$1
dim_y=$2

# Hardcoded output directory used in Python script.
output_directory=$PWD/output

# Parse arguments to remove those overwritten by this script.
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            # Using hardcoded directory so the user should not overwrite it.
            shift
            shift
            ;;
        *)
            args+=($1)
            shift
            ;;
    esac
done

./run_both.sh ${args[*]} -o ${output_directory}
if [ $? -ne 0 ]; then
    exit 1
fi

python=${root}/.venv/bin/python
${python} scripts/compare_densities.py ${dim_x} ${dim_y}
