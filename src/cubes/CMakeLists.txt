add_library(libcubes marching_cubes_grad.cpp marching_cubes_flat.cpp)
add_library(libcubesgen generator.cpp)

find_package(glm REQUIRED)
target_include_directories(libcubes PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(libcubesgen PUBLIC ${GLM_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CUBES_COM_OPT -march=native -ffast-math -funroll-loops)

    target_link_libraries(libcubes PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(libcubes PUBLIC ${CUBES_COM_OPT})    
    
    target_link_libraries(libcubesgen PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(libcubesgen PUBLIC ${CUBES_COM_OPT})
endif()

add_library(libcubescuda cuda_marching_cubes.cu)
set_property(TARGET libcubescuda PROPERTY CUDA_ARCHITECTURES 86)
target_include_directories(libcubescuda PUBLIC ${GLM_INCLUDE_DIRS})