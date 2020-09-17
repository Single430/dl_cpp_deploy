#!/bin/bash

rm -r build
mkdir build
cd build

cmake -DCMAKE_PREFIX_PATH=/绝对路径/src/libtorch  ..
cmake --build . --config Release --target demo -j 1
./demo ../mnist/model/mnist.pt ../mnist/model/keys.txt ../mnist/model/test.png
