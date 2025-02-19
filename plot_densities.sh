#!/bin/bash

./run_both.sh $1 $2 $3 $4 $5 $6 1
if [ $? -ne 0 ]; then
    exit 1
fi

python scripts/plot_densities.py $1 $2
