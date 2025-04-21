#include "marching_cubes_flat.hpp"
#include "marching_cubes_common.hpp"
#include <iostream>
#include <omp.h>
#include <chrono>

namespace MarchingCubesFlat
{
    inline int calcCubeIndex(GridCell &cell, float isovalue){
        int index = 0;
        for(int i = 0; i < 8; i++){
            if(cell[i].v < isovalue) index |= (1 << i);
        }
        return index;
    }

    inline glm::vec3 interpolate(GridCellEle &e1, GridCellEle &e2, float isovalue){
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        return glm::mix(e1.p, e2.p, mu);
    }
    
    inline std::vector<glm::vec3> intersection_coords(GridCell &cell, float isovalue, int cubeIndex){
        std::vector<glm::vec3> intersections(12);
    
        int intersectionsKey = edgeTable[cubeIndex];

        for(int idx = 0; intersectionsKey; idx++, intersectionsKey >>= 1){
            if(intersectionsKey & 1){
                glm::uvec2 v = edgeToVertices[idx];
                intersections[idx] = interpolate(cell[v.x], cell[v.y], isovalue);
            }
        }
    
        return intersections;
    }

    inline void triangles(std::vector<glm::vec3> &intersections, int cubeIndex,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        for(int i = 0; triTable[cubeIndex][i+2] != -1; i+=3){

            glm::vec3 p1 = intersections[triTable[cubeIndex][i]];
            glm::vec3 p2 = intersections[triTable[cubeIndex][i+1]];
            glm::vec3 p3 = intersections[triTable[cubeIndex][i+2]];
            outVerts.push_back(p1);
            outVerts.push_back(p2);
            outVerts.push_back(p3);
            glm::vec3 norm = glm::normalize(glm::cross(p2-p1,p3-p1));
            outNormals.push_back(norm);
            outNormals.push_back(norm);
            outNormals.push_back(norm);
        }
    }

    inline void trinagulate_cell(GridCell &cell, float isovalue,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        int cubeIndex = calcCubeIndex(cell, isovalue);
        std::vector<glm::vec3> intersections = intersection_coords(cell, isovalue, cubeIndex);
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
                            GridCellEle{ {p.x,      p.y,      p.z     }, grid(x,  y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z     }, grid(x+1,y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z+1.0f}, grid(x+1,y,  z+1) },
                            GridCellEle{ {p.x,      p.y,      p.z+1.0f}, grid(x,  y,  z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z     }, grid(x,  y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z     }, grid(x+1,y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z+1.0f}, grid(x+1,y+1,z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z+1.0f}, grid(x,  y+1,z+1) }
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
            std::mutex &mut, std::atomic_bool &should_stop, double delay){
            
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

                        if(should_stop.load(std::memory_order_relaxed))
                            continue;
                        
                        const glm::vec3 p(x,y,z);
    
                        GridCell cell{
                            GridCellEle{ {p.x,      p.y,      p.z     }, grid(x,  y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z     }, grid(x+1,y,  z  ) },
                            GridCellEle{ {p.x+1.0f, p.y,      p.z+1.0f}, grid(x+1,y,  z+1) },
                            GridCellEle{ {p.x,      p.y,      p.z+1.0f}, grid(x,  y,  z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z     }, grid(x,  y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z     }, grid(x+1,y+1,z  ) },
                            GridCellEle{ {p.x+1.0f, p.y+1.0f, p.z+1.0f}, grid(x+1,y+1,z+1) },
                            GridCellEle{ {p.x,      p.y+1.0f, p.z+1.0f}, grid(x,  y+1,z+1) }
                        };
    
                        trinagulate_cell(cell, isovalue, lVerts, lNorms);
                        
                        if(!lVerts.empty()){
                            if(delay > 0.0)
                                std::this_thread::sleep_for(std::chrono::duration<double>{delay});
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
