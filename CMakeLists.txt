cmake_minimum_required(VERSION 3.20)
project(marching_cubes)

add_executable(marching)
target_compile_features(marching PUBLIC cxx_std_20)

add_subdirectory(src)