add_library(libcubesgen generator.cpp)
add_library(libcubescpu CpuMC.cpp)

find_package(glm REQUIRED)
target_include_directories(libcubesgen PUBLIC ${GLM_INCLUDE_DIRS})
target_include_directories(libcubescpu PUBLIC ${GLM_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
set(CUBES_COM_OPT -march=native -ffast-math -funroll-loops)  

target_link_libraries(libcubesgen PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(libcubesgen PUBLIC ${CUBES_COM_OPT})

target_link_libraries(libcubescpu PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(libcubescpu PUBLIC ${CUBES_COM_OPT})
endif()

add_library(libcubescuda CudaMC.cu)
# target_compile_options(libcubescuda PUBLIC -Xptxas=-v)
set_property(TARGET libcubescuda PROPERTY CUDA_ARCHITECTURES native)
target_include_directories(libcubescuda PUBLIC ${GLM_INCLUDE_DIRS})