#!/bin/bash

cmake -S . -B build
cd build
make
./master_thesis $1 $2 $3 $4 $5 ../output
cd ..
