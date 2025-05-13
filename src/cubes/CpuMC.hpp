#pragma once
#include "grid.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <atomic>
#include <mutex>

namespace CpuMC{
    typedef std::pair<glm::vec3,glm::vec3> PG;
    typedef glm::vec3 P;

    template<typename T, bool use_omp>
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
    template<typename T, bool use_omp>
    void trinagulate_grid_mut(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay);
}