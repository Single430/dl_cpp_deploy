#!/bin/bash

rm -r build
mkdir build
cd build
cmake ..
make
./demo --source ../imgs/bus.jpg --weights ../weights/last.torchscript.pt --view-img
