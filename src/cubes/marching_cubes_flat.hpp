#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include "grid.hpp"

namespace MarchingCubesFlat
{
    struct GridCellEle{
        glm::vec3 p;
        float v;
    };
    using GridCell = std::array<GridCellEle,8>;

    int calcCubeIndex(GridCell &cell, float isovalue);

    glm::vec3 interpolate(GridCellEle &e1, GridCellEle &e2, float isovalue);

    std::vector<glm::vec3> intersection_coords(GridCell &cell, float isovalue, int cubeIndex);

    void triangles(std::vector<glm::vec3> &intersections, int cubeIndex,
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

    void trinagulate_cell(GridCell &cell, float isovalue,
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

    void triangulate_grid_mut(Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals, 
        std::mutex &mut, std::atomic_bool &should_stop);
}
