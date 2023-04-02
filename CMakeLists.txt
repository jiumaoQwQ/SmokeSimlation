cmake_minimum_required(VERSION 3.20)

project(SmokeSimulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp vdbhelper.cpp)

find_package(OpenVDB REQUIRED)

target_link_libraries(main OpenVDB::openvdb)

add_library(smoke STATIC SmokeSim.cu)

target_link_libraries(main smoke)