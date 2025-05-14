target_sources(marching PRIVATE marching.cpp)
target_sources(marching_vis PRIVATE marching_vis.cpp)

add_subdirectory(display)
target_link_libraries(marching_vis PUBLIC libdisplay)
target_include_directories(marching_vis PUBLIC display)

add_subdirectory(cubes)
target_link_libraries(marching PUBLIC libcubesgen)
target_link_libraries(marching PUBLIC libcubescuda)
target_link_libraries(marching PUBLIC libcubescpu)
target_include_directories(marching PUBLIC cubes)

target_link_libraries(marching_vis PUBLIC libcubesgen)
target_link_libraries(marching_vis PUBLIC libcubescpu)
target_link_libraries(marching_vis PUBLIC libcubescuda)
target_include_directories(marching_vis PUBLIC cubes)

find_package(glm REQUIRED)
target_include_directories(marching PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(marching_vis PUBLIC ${GLM_INCLUDE_DIRS})

# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
#     set(MARCH_COM_OPT -march=native -ffast-math -funroll-loops)
#     target_link_libraries(marching PUBLIC OpenMP::OpenMP_CXX)
#     target_compile_options(marching PUBLIC ${MARCH_COM_OPT})
    
#     target_link_libraries(marching_vis PUBLIC OpenMP::OpenMP_CXX)
#     target_compile_options(marching_vis PUBLIC ${MARCH_COM_OPT})
# endif()