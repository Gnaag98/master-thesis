#!/bin/bash

cmake -S . -B build
cd build
make
./master_thesis $1 $2 $3 $4 $5 0 ../output $6 $7
./master_thesis $1 $2 $3 $4 $5 1 ../output $6 $7
cd ..
python scripts/compare_densities.py $1 $2
