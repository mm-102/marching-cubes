cmake_minimum_required(VERSION 3.20)
project(marching_cubes)

add_executable(marching)

find_package(OpenGL REQUIRED)
target_include_directories(marching PUBLIC ${OPENGL_INCLUDE_DIR})
target_link_libraries(marching PUBLIC ${OPENGL_LIBRARIES})

find_package(glm REQUIRED)
target_include_directories(marching PUBLIC ${GLM_INCLUDE_DIRS})

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(marching PUBLIC glfw)

find_package(GLEW 2.2 REQUIRED)
target_link_libraries(marching PUBLIC GLEW::GLEW)

add_subdirectory(src)