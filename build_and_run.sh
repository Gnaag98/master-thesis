#!/bin/bash

cmake -S . -B build
cd build
make
./master_thesis $1 ../output
cd ..
