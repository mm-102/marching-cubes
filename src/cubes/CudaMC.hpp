#pragma once
#include "grid.hpp"
#include <vector_types.h>
#include <vector_functions.hpp>
#include <vector>
#include <tuple>

namespace CudaMC{
    struct PG{
        float3 p;
        float3 g;
    };

    typedef float3 P;

    template<typename T>
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
    void setConstMem();
    std::tuple<int64_t,int64_t,int64_t> test_time(const Grid<float> &grid, float isovalue);
}