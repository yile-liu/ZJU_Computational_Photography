cmake_minimum_required(VERSION 3.13)
project(ImageCompletion)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(ImageCompletion main.cpp PointManager.cpp PointManager.h
        StructurePropagation.cpp StructurePropagation.h PhotometricalCorrection.cpp PhotometricalCorrection.h TexturePropagation.cpp
        TexturePropagation.h)


target_link_libraries(ImageCompletion ${OpenCV_LIBS})