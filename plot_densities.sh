#!/bin/bash

./run_both.sh $1 $2 $3 $4 $5 $6 1

python scripts/plot_densities.py $1 $2 $7
