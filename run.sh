#!/bin/bash

rm -r build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/run/media/zbl/works/c.c++/src/libtorch  ..
cmake --build . --config Release --target demo -j 1
#./demo /run/media/zbl/works/c.c++/crnn_libtorch/src/crnn.pt /run/media/zbl/works/c.c++/crnn_libtorch/src/keys.txt /run/media/zbl/works/c.c++/crnn_libtorch/src/test.jpg
./demo ../mnist/model/mnist.pt ../mnist/model/keys.txt ../mnist/model/test.png
