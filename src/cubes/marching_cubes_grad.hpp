#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <array>
#include <mutex>
#include <thread>
#include <atomic>
#include "grid.hpp"

namespace MarchingCubesGrad
{
    struct GridCellEle{
        glm::vec3 p;
        glm::vec3 g;
        float v;
    };
    using GridCell = std::array<GridCellEle,8>;
    using PosGrad = std::pair<glm::vec3,glm::vec3>;

    int calcCubeIndex(GridCell &cell, float isovalue);

    glm::vec3 calcGrad(const Grid<float> &grid, int x, int y, int z);

    PosGrad interpolate(GridCellEle &e1, GridCellEle &e2, float isovalue);

    std::vector<PosGrad> intersection_coords(GridCell &cell, float isovalue, int cubeIndex);

    void triangles(std::vector<PosGrad> &intersections, int cubeIndex,
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

    void trinagulate_cell(GridCell &cell, float isovalue,
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

    void triangulate_grid_mut(Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals, 
        std::mutex &mut, std::atomic_bool &should_stop, double delay);
}
