cmake_minimum_required(VERSION 3.5.1)
project(face_detector_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/include/ncnn
)

link_directories(
        ${OpenCV_DIR}/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib
)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ncnn
        libgomp.so.1
)
