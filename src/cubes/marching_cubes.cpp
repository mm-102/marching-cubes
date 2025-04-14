#include "marching_cubes.hpp"

namespace MarchingCubes
{
    int calcCubeIndex(GridCell &cell, float isovalue){
        int index = 0;
        for(int i = 0; i < 8; i++){
            if(cell.v[i].w < isovalue) index |= (1 << i);
        }
        return index;
    }
    
    glm::vec3 interpolate(glm::vec4 v1, glm::vec4 v2, float isovalue){
        float mu = (isovalue - v1.w) / (v2.w - v1.w);
        return glm::vec3(glm::mix(v1, v2, mu));
    }
    
    std::vector<glm::vec3> intersection_coords(GridCell &cell, float isovalue, int cubeIndex){
        std::vector<glm::vec3> intersections(12);
    
        int intersectionsKey = edgeTable[cubeIndex];
    
        for(int idx = 0; intersectionsKey; idx++, intersectionsKey >>= 1){
            if(intersectionsKey & 1){
                glm::uvec2 v = edgeToVertices[idx];
                intersections[idx] = interpolate(cell.v[v.x], cell.v[v.y], isovalue);
            }
        }
    
        return intersections;
    }
    
    std::vector<glm::vec3> triangles(std::vector<glm::vec3> &intersections, int cubeIndex){
        std::vector<glm::vec3> triangles;
        for(int i = 0; MarchingCubes::triTable[cubeIndex][i] != -1; i+=3){
            triangles.push_back(intersections[triTable[cubeIndex][i]]);
            triangles.push_back(intersections[triTable[cubeIndex][i+1]]);
            triangles.push_back(intersections[triTable[cubeIndex][i+2]]);
        }
        return triangles;
    }
    
    std::vector<glm::vec3> trinagulate_cell(GridCell &cell, float isovalue){
        int cubeIndex = calcCubeIndex(cell, isovalue);
        std::vector<glm::vec3> intersections = intersection_coords(cell, isovalue, cubeIndex);
        return triangles(intersections, cubeIndex);
    }
    
    std::vector<glm::vec3> trinagulate_grid(Grid<float> &grid, float isovalue){
        std::vector<glm::vec3> triangles;
    
        glm::uvec3 grid_size = grid.getSize();
    
        for(int z = 0; z + 1 < grid_size.z; z++){
            for(int y = 0; y + 1 < grid_size.y; y++){
                for(int x = 0; x + 1 < grid_size.x; x++){
                    
                    glm::vec3 p(x,y,z);
    
                    MarchingCubes::GridCell cell{ .v{
                        {p.x,      p.y,      p.z,      grid(x,  y  ,z  )},
                        {p.x+1.0f, p.y,      p.z,      grid(x+1,y  ,z  )},
                        {p.x+1.0f, p.y,      p.z+1.0f, grid(x+1,y  ,z+1)},
                        {p.x,      p.y,      p.z+1.0f, grid(x  ,y  ,z+1)},
                        {p.x,      p.y+1.0f, p.z,      grid(x  ,y+1,z  )},
                        {p.x+1.0f, p.y+1.0f, p.z,      grid(x+1,y+1,z  )},
                        {p.x+1.0f, p.y+1.0f, p.z+1.0f, grid(x+1,y+1,z+1)},
                        {p.x,      p.y+1.0f, p.z+1.0f, grid(x  ,y+1,z+1)}
                    }};
    
                    std::vector<glm::vec3> cell_trinagles = MarchingCubes::trinagulate_cell(cell, isovalue);
                    triangles.reserve(triangles.size() + cell_trinagles.size());
                    triangles.insert(triangles.end(), cell_trinagles.begin(), cell_trinagles.end());
                }
            }
        }
    
        return triangles;
    }
}
