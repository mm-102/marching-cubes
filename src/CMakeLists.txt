target_sources(marching PRIVATE marching.cpp)
target_sources(marching_vis PRIVATE marching_vis.cpp)

add_subdirectory(display)
target_link_libraries(marching_vis PUBLIC libdisplay)
target_include_directories(marching_vis PUBLIC display)

add_subdirectory(cubes)
target_link_libraries(marching PUBLIC libcubesgrad)
target_link_libraries(marching PUBLIC libcubesflat)
target_link_libraries(marching PUBLIC libcubesgen)
target_include_directories(marching PUBLIC cubes)

target_link_libraries(marching_vis PUBLIC libcubesgrad)
target_link_libraries(marching_vis PUBLIC libcubesflat)
target_link_libraries(marching_vis PUBLIC libcubesgen)
target_include_directories(marching_vis PUBLIC cubes)

find_package(glm REQUIRED)
target_include_directories(marching_vis PUBLIC ${GLM_INCLUDE_DIRS})