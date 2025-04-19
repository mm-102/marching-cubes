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
            if(cell.v[i].w < isovalue) index |= (1 << i);
        }
        return index;
    }
    
    inline glm::vec3 interpolate(glm::vec4 v1, glm::vec4 v2, float isovalue){
        float mu = (isovalue - v1.w) / (v2.w - v1.w);
        return glm::vec3(glm::mix(v1, v2, mu));
    }
    
    inline std::vector<glm::vec3> intersection_coords(GridCell &cell, float isovalue, int cubeIndex){
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
    
    inline std::vector<glm::vec3> triangles(std::vector<glm::vec3> &intersections, int cubeIndex){
        std::vector<glm::vec3> triangles;
        for(int i = 0; MarchingCubes::triTable[cubeIndex][i] != -1; i++){
            triangles.push_back(intersections[triTable[cubeIndex][i]]);
        }
        return triangles;
    }
    
    inline std::vector<glm::vec3> trinagulate_cell(GridCell &cell, float isovalue){
        int cubeIndex = calcCubeIndex(cell, isovalue);
        std::vector<glm::vec3> intersections = intersection_coords(cell, isovalue, cubeIndex);
        return triangles(intersections, cubeIndex);
    }
    
    std::vector<glm::vec3> trinagulate_grid(Grid<float> &grid, float isovalue){
        #pragma omp declare reduction(merge : std::vector<glm::vec3> : \
            omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

        const glm::uvec3 grid_size = grid.getSize();
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 9;
        
        std::vector<glm::vec3> triangles;
        triangles.reserve(res);
        const int max_z = grid_size.z - 1;
        
        #pragma omp parallel for reduction(merge: triangles)
        for(int z = 0; z < max_z; z++){
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
                    if(triangles.capacity() < triangles.size() + cell_trinagles.size()){
                        triangles.reserve(triangles.capacity() + res / omp_get_num_threads());
                    }
                    triangles.reserve(triangles.size() + cell_trinagles.size());
                    triangles.insert(triangles.end(), cell_trinagles.begin(), cell_trinagles.end());
                }
            }
        }
    
        return triangles;
    }
    
    void triangulate_grid_to_vec(Grid<float> &grid, float isovalue, std::vector<glm::vec3> &vec, std::mutex &mut, std::atomic_bool &should_stop){        
        const glm::uvec3 grid_size = grid.getSize();
        const int max_z = grid_size.z - 1;
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 10;

        #pragma omp parallel for schedule(static)
        for(int z = 0; z < max_z; z++){
            for(int y = 0; y + 1 < grid_size.y; y++){
                for(int x = 0; x + 1 < grid_size.x; x++){
                    
                    if(should_stop){ // kind of janky
                        y = grid_size.y;
                        x = grid_size.x;
                        break;
                    }

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
                    if(!cell_trinagles.empty()){
                        // std::this_thread::sleep_for(std::chrono::duration<double>{0.0001});
                        std::lock_guard<std::mutex> lock(mut);
                        if(vec.capacity() < vec.size() + cell_trinagles.size()){
                            vec.reserve(vec.capacity() + res);
                        }
                        vec.reserve(vec.size() + cell_trinagles.size());  
                        vec.insert(vec.end(), cell_trinagles.begin(), cell_trinagles.end());
                    }
                }
            }
        }
    }
}
