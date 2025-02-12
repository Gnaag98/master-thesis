#!/bin/bash

dim_x=$1
dim_y=$2
dim_z=$3
cell_size=$4
particles_per_cell=$5
distribution=$6
should_save=$7
filepath=$8

./build.sh
if [ $? -ne 0 ]; then
    exit 1
fi

cd build
./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} ${particles_per_cell} \
    0 ../output ${distribution} ${should_save} ../${filepath}
if [ $? -ne 0 ]; then
    cd ..
    exit 1
fi
./master_thesis ${dim_x} ${dim_y} ${dim_z} ${cell_size} ${particles_per_cell} \
    1 ../output ${distribution} ${should_save} ../${filepath}
if [ $? -ne 0 ]; then
    cd ..
    exit 1
fi
cd ..
