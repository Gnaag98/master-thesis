#!/bin/bash

cmake -S . -B build
cd build
make
if [ $? -ne 0 ]; then
    exit 1
fi
cd ..
