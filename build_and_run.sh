#!/bin/bash

cmake -S . -B build
cd build
make
if [ $? -ne 0 ]; then
    return 1
fi
./master_thesis $1 $2 $3 $4 $5 0 ../output $6 $7
if [ $? -ne 0 ]; then
    cd ..
    return 1
fi
./master_thesis $1 $2 $3 $4 $5 1 ../output $6 $7
if [ $? -ne 0 ]; then
    cd ..
    return 1
fi
cd ..
python scripts/compare_densities.py $1 $2
