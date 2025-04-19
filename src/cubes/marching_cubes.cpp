#include "marching_cubes.hpp"
#include <iostream>
#include <omp.h>
#include <chrono>
#include <functional>
namespace MarchingCubes
{
    inline int calcCubeIndex(GridCell &cell, float isovalue){
        int index = 0;
        for(int i = 0; i < 8; i++){
            if(cell[i].v < isovalue) index |= (1 << i);
        }
        return index;
    }

    inline glm::vec3 calcGrad(const Grid<float> &grid, int x, int y, int z){
        glm::ivec3 size = glm::ivec3(grid.getSize()) - 1;

        int xm = glm::clamp(x - 1, 0, size.x);
        int xp = glm::clamp(x + 1, 0, size.x);
        int ym = glm::clamp(y - 1, 0, size.y);
        int yp = glm::clamp(y + 1, 0, size.y);
        int zm = glm::clamp(z - 1, 0, size.z);
        int zp = glm::clamp(z + 1, 0, size.z);

        float dx = grid(xp, y, z) - grid(xm, y, z);
        float dy = grid(x, yp, z) - grid(x, ym, z);
        float dz = grid(x, y, zp) - grid(x, y, zm);

        return glm::normalize(glm::vec3(dx, dy, dz));
    }

    inline PosGrad interpolate(GridCellEle &e1, GridCellEle &e2, float isovalue){
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        glm::vec3 pos = glm::mix(e1.p, e2.p, mu);
        glm::vec3 grad = glm::normalize(glm::mix(e1.g, e2.g, mu));
        return {pos, grad};
    }
    
    inline std::vector<PosGrad> intersection_coords(GridCell &cell, float isovalue, int cubeIndex){
        std::vector<PosGrad> intersections(12);
    
        int intersectionsKey = edgeTable[cubeIndex];

        for(int idx = 0; intersectionsKey; idx++, intersectionsKey >>= 1){
            if(intersectionsKey & 1){
                glm::uvec2 v = edgeToVertices[idx];
                intersections[idx] = interpolate(cell[v.x], cell[v.y], isovalue);
            }
        }
    
        return intersections;
    }

    inline void triangles(std::vector<PosGrad> &intersections, int cubeIndex,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        for(int i = 0; triTable[cubeIndex][i] != -1; i++){
            auto [pos, norm] = intersections[triTable[cubeIndex][i]];
            outVerts.push_back(pos);
            outNormals.push_back(norm);
        }
    }

    inline void trinagulate_cell(GridCell &cell, float isovalue,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        int cubeIndex = calcCubeIndex(cell, isovalue);
        std::vector<PosGrad> intersections = intersection_coords(cell, isovalue, cubeIndex);
        triangles(intersections,cubeIndex,outVerts,outNormals);
    }
    
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        const glm::uvec3 grid_size = grid.getSize();
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 9;
        
        outVerts.reserve(res);
        outNormals.reserve(res);
        
        #pragma omp parallel
        {
            std::vector<glm::vec3> lVerts, lNorms;
            lVerts.reserve(2 * res / omp_get_num_threads());
            lNorms.reserve(2 * res / omp_get_num_threads());

            #pragma omp for collapse(3) schedule(static) nowait
            for(int z = 0; z < grid_size.z - 1; z++){
                for(int y = 0; y < grid_size.y - 1; y++){
                    for(int x = 0; x < grid_size.x - 1; x++){
                        
                        const glm::vec3 p(x,y,z);
    
                        GridCell cell{
                            GridCellEle{ {p.x,      p.y,      p.z     }, calcGrad(grid, x,  y,  z  ), grid(x,  y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z     }, calcGrad(grid, x+1,y,  z  ), grid(x+1,y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z+1.0f}, calcGrad(grid, x+1,y,  z+1), grid(x+1,y,  z+1) },
                            GridCellEle{ {p.x,      p.y,      p.z+1.0f}, calcGrad(grid, x,  y,  z+1), grid(x,  y,  z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z     }, calcGrad(grid, x,  y+1,z  ), grid(x,  y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z     }, calcGrad(grid, x+1,y+1,z  ), grid(x+1,y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z+1.0f}, calcGrad(grid, x+1,y+1,z+1), grid(x+1,y+1,z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z+1.0f}, calcGrad(grid, x,  y+1,z+1), grid(x,  y+1,z+1) }
                        };
    
                        trinagulate_cell(cell, isovalue, lVerts, lNorms);
                        
                    }
                }
            }

            #pragma omp critical
            {
                outVerts.insert(outVerts.end(), lVerts.begin(), lVerts.end());
                outNormals.insert(outNormals.end(), lNorms.begin(), lNorms.end());
            }
        }


    }
    
    void triangulate_grid_mut(Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals, 
            std::mutex &mut, std::atomic_bool &should_stop){
            
        const glm::uvec3 grid_size = grid.getSize();
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 9;
        {
            std::lock_guard<std::mutex> lock(mut);
            outVerts.reserve(res);
            outNormals.reserve(res);
        }

        #pragma omp parallel
        {
            std::vector<glm::vec3> lVerts, lNorms;

            #pragma omp for collapse(3) schedule(static) nowait
            for(int z = 0; z < grid_size.z - 1; z++){
                for(int y = 0; y < grid_size.y - 1; y++){
                    for(int x = 0; x < grid_size.x - 1; x++){
                        
                        const glm::vec3 p(x,y,z);
    
                        GridCell cell{
                            GridCellEle{ {p.x,      p.y,      p.z     }, calcGrad(grid, x,  y,  z  ), grid(x,  y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z     }, calcGrad(grid, x+1,y,  z  ), grid(x+1,y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z+1.0f}, calcGrad(grid, x+1,y,  z+1), grid(x+1,y,  z+1) },
                            GridCellEle{ {p.x,      p.y,      p.z+1.0f}, calcGrad(grid, x,  y,  z+1), grid(x,  y,  z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z     }, calcGrad(grid, x,  y+1,z  ), grid(x,  y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z     }, calcGrad(grid, x+1,y+1,z  ), grid(x+1,y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z+1.0f}, calcGrad(grid, x+1,y+1,z+1), grid(x+1,y+1,z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z+1.0f}, calcGrad(grid, x,  y+1,z+1), grid(x,  y+1,z+1) }
                        };
    
                        trinagulate_cell(cell, isovalue, lVerts, lNorms);
                        
                        if(!lVerts.empty()){
                            std::lock_guard<std::mutex> lock(mut);
                            outVerts.insert(outVerts.end(), lVerts.begin(), lVerts.end());
                            outNormals.insert(outNormals.end(), lNorms.begin(), lNorms.end());
                        }
                        lVerts.clear();
                        lNorms.clear();
                    }
                }
            }
        }

    }
}
