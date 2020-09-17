#!/bin/bash

rm -r build
mkdir build
cd build
<<<<<<< HEAD
cmake -DCMAKE_PREFIX_PATH=/run/media/zbl/works/c.c++/src/libtorch  ..
cmake --build . --config Release --target mnist -j 1
#./demo /run/media/zbl/works/c.c++/crnn_libtorch/src/crnn.pt /run/media/zbl/works/c.c++/crnn_libtorch/src/keys.txt /run/media/zbl/works/c.c++/crnn_libtorch/src/test.jpg
./mnist ../mnist/model/mnist.pt ../mnist/model/keys.txt ../mnist/model/test.png
=======
cmake -DCMAKE_PREFIX_PATH=/绝对路径/src/libtorch  ..
cmake --build . --config Release --target demo -j 1
./demo ../mnist/model/mnist.pt ../mnist/model/keys.txt ../mnist/model/test.png
>>>>>>> 9cce0b26d4f542690fb2595706a8134daea03842
