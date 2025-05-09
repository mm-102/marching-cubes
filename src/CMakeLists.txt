target_sources(marching PRIVATE marching.cu)
target_sources(marching_vis PRIVATE marching_vis.cpp)

add_subdirectory(display)
target_link_libraries(marching_vis PUBLIC libdisplay)
target_include_directories(marching_vis PUBLIC display)

add_subdirectory(cubes)
target_link_libraries(marching PUBLIC libcubes)
target_link_libraries(marching PUBLIC libcubesgen)
# target_link_libraries(marching PUBLIC libcubescuda) # temp
target_include_directories(marching PUBLIC cubes)

target_link_libraries(marching_vis PUBLIC libcubes)
target_link_libraries(marching_vis PUBLIC libcubesgen)
target_include_directories(marching_vis PUBLIC cubes)

find_package(glm REQUIRED)
target_include_directories(marching_vis PUBLIC ${GLM_INCLUDE_DIRS})