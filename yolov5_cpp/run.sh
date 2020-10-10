#!/bin/bash

rm -r build
mkdir build
cd build
cmake ..
make
./demo --img ../imgs/bus.jpg --weights ../weights/last.torchscript.pt --view-img
