cmake_minimum_required(VERSION 3.20)
project(marching_cubes)
enable_language(CUDA)

add_executable(marching)
add_executable(marching_vis)

add_executable(time_test)

add_subdirectory(src)