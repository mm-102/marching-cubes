file(GLOB cubes_src *.cpp)
add_library(libcubes ${cubes_src})

find_package(glm REQUIRED)
target_include_directories(libcubes PUBLIC ${GLM_INCLUDE_DIRS})