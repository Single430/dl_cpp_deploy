#!/bin/bash

rm -r build
mkdir build
cd build
cmake ..
make

./face_detector_cpp --img ../sample.jpg --param ../model/mobile0.25.param --bin ../model/mobile0.25.bin
