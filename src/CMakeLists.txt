target_sources(marching PRIVATE main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(marching PUBLIC OpenMP::OpenMP_CXX)
endif()

add_subdirectory(display)
target_link_libraries(marching PUBLIC libdisplay)
target_include_directories(marching PUBLIC display)

add_subdirectory(cubes)
target_link_libraries(marching PUBLIC libcubes)
target_include_directories(marching PUBLIC cubes)