file(GLOB cubes_src *.cpp)
add_library(libcubes ${cubes_src})

find_package(glm REQUIRED)
target_include_directories(libcubes PUBLIC ${GLM_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(libcubes PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(libcubes PUBLIC
        -march=native
        -ffast-math
        -funroll-loops
    )
endif()