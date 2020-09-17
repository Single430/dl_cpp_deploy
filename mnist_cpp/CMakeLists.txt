# cmake needs this line
cmake_minimum_required(VERSION 3.15)

# Enable C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(mnist)

#include_directories(../src/libtorch/include)
#include_directories(../src/libtorch/include/torch/csrc/api/include)
#include_directories(/usr/include/opencv4)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

# Print some message showing some of them
message(STATUS "Pytorch status:")
message(STATUS "	libraries: ${TORCH_LIBARRIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_executable(mnist main.cpp MnistFunc.h)
target_link_libraries(mnist "${TORCH_LIBRARIES}")
target_link_libraries(mnist ${OpenCV_LIBS})
set_property(TARGET mnist PROPERTY CXX_STANDARD 14)
