add_library(libcubesgrad marching_cubes_grad.cpp)
add_library(libcubesflat marching_cubes_flat.cpp)
add_library(libcubesgen generator.cpp)

find_package(glm REQUIRED)
target_include_directories(libcubesgrad PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(libcubesflat PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(libcubesgen PUBLIC ${GLM_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CUBES_COM_OPT -march=native -ffast-math -funroll-loops)

    target_link_libraries(libcubesgrad PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(libcubesflat PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(libcubesgen PUBLIC ${CUBES_COM_OPT})    
    
    target_link_libraries(libcubesgrad PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(libcubesflat PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(libcubesgen PUBLIC ${CUBES_COM_OPT})
endif()