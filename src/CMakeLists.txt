target_sources(marching PRIVATE main.cpp)

add_subdirectory(display)
target_link_libraries(marching PUBLIC libdisplay)
target_include_directories(marching PUBLIC display)