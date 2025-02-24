#!/bin/bash

./build.sh
if [ $? -ne 0 ]; then
    exit 1
fi

# Make sure both versions use the same seed.
random_seed=$RANDOM

# Parse arguments to remove those overwritten by this script.
args=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -r)
            # Replace generated seed with user defined seed.
            random_seed=$2
            shift
            shift
            ;;
        -v)
            # No reason to specify a version when both version will be run.
            shift
            shift
            ;;
        *)
            args+=($1)
            shift
            ;;
    esac
done

build/master_thesis ${args[*]} -r ${random_seed} -v global
if [ $? -ne 0 ]; then
    exit 1
fi

build/master_thesis ${args[*]} -r ${random_seed} -v shared
if [ $? -ne 0 ]; then
    exit 1
fi
