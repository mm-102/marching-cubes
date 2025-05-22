#include "CpuMC.hpp"
#include "common.hpp"
#include <array>
#include <cstdint>
#include <omp.h>
#include <chrono>
#include <thread>

namespace CpuMC{

    template<typename T>
    struct Ele{
        T d;
        float v;
    };

    template<typename T>
    using GridCell = std::array<T,8>;

    template<typename T>
    inline uint8_t calcCubeIndex(GridCell<T> &cell, const float &isovalue){
        uint8_t index = 0;
        if(cell[0].v < isovalue) index |= 1;
        if(cell[1].v < isovalue) index |= 2;
        if(cell[2].v < isovalue) index |= 4;
        if(cell[3].v < isovalue) index |= 8;
        if(cell[4].v < isovalue) index |= 16;
        if(cell[5].v < isovalue) index |= 32;
        if(cell[6].v < isovalue) index |= 64;
        if(cell[7].v < isovalue) index |= 128;
        return index;
    }

    inline glm::vec3 calcGrad(const Grid<float> &grid, const int &x, const int &y, const int &z){        
        // glm::uvec3 s = grid.getSize();
        // if(x < 1 || y < 1 || z < 1 || x > s.x - 2 || y > s.y - 2 || z > s.z - 2)
        //     return glm::vec3(0);

        glm::uvec3 p = glm::clamp(glm::uvec3(x,y,z), glm::uvec3(1), grid.getSize() - 2u);
        
        return glm::normalize(glm::vec3(
            grid(p.x + 1, p.y, p.z) - grid(p.x - 1, p.y, p.z),
            grid(p.x, p.y + 1, p.z) - grid(p.x, p.y - 1, p.z),
            grid(p.x, p.y, p.z + 1) - grid(p.x, p.y, p.z - 1)
        ));
    }

    template<typename T>
    void interpolate(const Ele<T>& e1, const Ele<T>& e2, const float &isovalue, T &out);

    template<> inline void interpolate<P>(const Ele<P>& e1, const Ele<P>& e2, const float &isovalue, P &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out = mix(e1.d, e2.d, mu);
    }

    template<> inline void interpolate<PG>(const Ele<PG>& e1, const Ele<PG>& e2, const float &isovalue, PG &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out.first = glm::mix(e1.d.first, e2.d.first, mu);
        out.second = glm::normalize(glm::mix(e1.d.second, e2.d.second, mu));
    }

    template<typename T>
    inline void triangles(std::array<T,12> &intersections, const int &cubeIndex,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
    template<> inline void triangles(std::array<P,12> &intersections, const int &cubeIndex,
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

    template<> inline void triangles(std::array<PG,12> &intersections, const int &cubeIndex,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        for(int i = 0; triTable[cubeIndex][i] != -1; i++){
            auto [pos, norm] = intersections[triTable[cubeIndex][i]];
            outVerts.push_back(pos);
            outNormals.push_back(norm);
        }
    }

    template<typename T>
    inline void trinagulate_cell(GridCell<Ele<T>> &cell, float isovalue,
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){
        
        uint8_t cubeIndex = calcCubeIndex<Ele<T>>(cell, isovalue);

        if(edgeTable[cubeIndex] == 0) return;

        std::array<T,12> intersections;
        uint16_t intersectionsKey = edgeTable[cubeIndex];
        glm::uvec2 v;
        for(int idx = 0; intersectionsKey; idx++, intersectionsKey >>= 1) {
            if (intersectionsKey & 1) {
                v = edgeToVertices[idx];
                interpolate<T>(cell[v.x], cell[v.y], isovalue, intersections[idx]);
            }
        }
        triangles<T>(intersections, cubeIndex, outVerts, outNormals); 
    }

    template<typename T>
    inline GridCell<T> make_cell(const Grid<float> &grid, const glm::vec3 &p, const int &x, const int &y, const int &z);

    template<> inline GridCell<Ele<P>> make_cell(const Grid<float> &grid, const glm::vec3 &p, const int &x, const int &y, const int &z){
        return GridCell<Ele<P>>{
            Ele<P>{ {p.x,      p.y,      p.z     }, grid(x,  y,  z  ) },
            Ele<P>{ {p.x+1.0f, p.y,      p.z     }, grid(x+1,y,  z  ) },
            Ele<P>{ {p.x+1.0f, p.y,      p.z+1.0f}, grid(x+1,y,  z+1) },
            Ele<P>{ {p.x,      p.y,      p.z+1.0f}, grid(x,  y,  z+1) },
            Ele<P>{ {p.x,      p.y+1.0f, p.z     }, grid(x,  y+1,z  ) },
            Ele<P>{ {p.x+1.0f, p.y+1.0f, p.z     }, grid(x+1,y+1,z  ) },
            Ele<P>{ {p.x+1.0f, p.y+1.0f, p.z+1.0f}, grid(x+1,y+1,z+1) },
            Ele<P>{ {p.x,      p.y+1.0f, p.z+1.0f}, grid(x,  y+1,z+1) }
        };
    }

    template<> inline GridCell<Ele<PG>> make_cell(const Grid<float> &grid, const glm::vec3 &p, const int &x, const int &y, const int &z){
        return GridCell<Ele<PG>>{
            Ele<PG>{ PG{{p.x,      p.y,      p.z     }, calcGrad(grid, x,  y,  z  )}, grid(x,  y,  z  ) },
            Ele<PG>{ PG{{p.x+1.0f, p.y,      p.z     }, calcGrad(grid, x+1,y,  z  )}, grid(x+1,y,  z  ) },
            Ele<PG>{ PG{{p.x+1.0f, p.y,      p.z+1.0f}, calcGrad(grid, x+1,y,  z+1)}, grid(x+1,y,  z+1) },
            Ele<PG>{ PG{{p.x,      p.y,      p.z+1.0f}, calcGrad(grid, x,  y,  z+1)}, grid(x,  y,  z+1) },
            Ele<PG>{ PG{{p.x,      p.y+1.0f, p.z     }, calcGrad(grid, x,  y+1,z  )}, grid(x,  y+1,z  ) },
            Ele<PG>{ PG{{p.x+1.0f, p.y+1.0f, p.z     }, calcGrad(grid, x+1,y+1,z  )}, grid(x+1,y+1,z  ) },
            Ele<PG>{ PG{{p.x+1.0f, p.y+1.0f, p.z+1.0f}, calcGrad(grid, x+1,y+1,z+1)}, grid(x+1,y+1,z+1) },
            Ele<PG>{ PG{{p.x,      p.y+1.0f, p.z+1.0f}, calcGrad(grid, x,  y+1,z+1)}, grid(x,  y+1,z+1) }
        };
    }

    template<typename T, bool use_omp>
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        const glm::uvec3 grid_size = grid.getSize();
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 9;
        
        outVerts.reserve(res);
        outNormals.reserve(res);

        if constexpr (use_omp){

            #pragma omp parallel
            {
                std::vector<glm::vec3> lVerts, lNorms;
                lVerts.reserve(2 * res / omp_get_num_threads());
                lNorms.reserve(2 * res / omp_get_num_threads());
    
                #pragma omp for collapse(3) schedule(static) nowait
                for(int z = 0; z < grid_size.z - 1; z++){
                    for(int y = 0; y < grid_size.y - 1; y++){
                        for(int x = 0; x < grid_size.x - 1; x++){

                            if(lVerts.capacity() < lVerts.size() + 15){
                                size_t res = static_cast<size_t>(1.5f * lVerts.size());
                                lVerts.reserve(res);
                                lNorms.reserve(res);
                            }
                            
                            const glm::vec3 p(x,y,z);
                            GridCell<Ele<T>> cell = make_cell<Ele<T>>(grid, p, x, y, z);
                            trinagulate_cell<T>(cell, isovalue, lVerts, lNorms);
                        }
                    }
                }
    
                #pragma omp critical
                {
                    outVerts.reserve(outVerts.size() + lVerts.size());
                    outNormals.reserve(outNormals.size() + lNorms.size());
                    outVerts.insert(outVerts.end(), lVerts.begin(), lVerts.end());
                    outNormals.insert(outNormals.end(), lNorms.begin(), lNorms.end());
                }
            }
        }
        else {
            for(int z = 0; z < grid_size.z - 1; z++){
                for(int y = 0; y < grid_size.y - 1; y++){
                    for(int x = 0; x < grid_size.x - 1; x++){
                        
                        const glm::vec3 p(x,y,z);
                        GridCell<Ele<T>> cell = make_cell<Ele<T>>(grid, p, x, y, z);
                        trinagulate_cell<T>(cell, isovalue, outVerts, outNormals);
                    }
                }
            }
        }
        
    }

    template<typename T, bool use_omp>
    void trinagulate_grid_mut(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay){

        const glm::uvec3 grid_size = grid.getSize();
        const unsigned res = grid_size.x * grid_size.y * grid_size.z / 9;
        
        {
            std::lock_guard<std::mutex> lock(mut);
            outVerts.reserve(res);
            outNormals.reserve(res);
        }

        if constexpr (use_omp){

            #pragma omp parallel
            {
                std::vector<glm::vec3> lVerts, lNorms;
                lVerts.reserve(res / omp_get_num_threads());
                lNorms.reserve(res / omp_get_num_threads());
    
                #pragma omp for collapse(3) schedule(static) nowait
                for(int z = 0; z < grid_size.z - 1; z++){
                    for(int y = 0; y < grid_size.y - 1; y++){
                        for(int x = 0; x < grid_size.x - 1; x++){

                            if(should_stop.load(std::memory_order_relaxed))
                                continue;
                            
                            const glm::vec3 p(x,y,z);
                            GridCell<Ele<T>> cell = make_cell<Ele<T>>(grid, p, x, y, z);
                            trinagulate_cell<T>(cell, isovalue, lVerts, lNorms);

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
        else {
            std::vector<glm::vec3> lVerts, lNorms;
            for(int z = 0; z < grid_size.z - 1; z++){
                for(int y = 0; y < grid_size.y - 1; y++){
                    for(int x = 0; x < grid_size.x - 1; x++){

                        if(should_stop.load(std::memory_order_relaxed))
                            continue;
                        
                        const glm::vec3 p(x,y,z);
                        GridCell<Ele<T>> cell = make_cell<Ele<T>>(grid, p, x, y, z);
                        trinagulate_cell<T>(cell, isovalue, lVerts, lNorms);

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

template void CpuMC::trinagulate_grid<CpuMC::P,true>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

template void CpuMC::trinagulate_grid<CpuMC::P,false>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

template void CpuMC::trinagulate_grid<CpuMC::PG,true>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

template void CpuMC::trinagulate_grid<CpuMC::PG,false>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

template void CpuMC::trinagulate_grid_mut<CpuMC::P,true>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay);

template void CpuMC::trinagulate_grid_mut<CpuMC::P,false>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay);
            
template void CpuMC::trinagulate_grid_mut<CpuMC::PG,true>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay);

template void CpuMC::trinagulate_grid_mut<CpuMC::PG,false>(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals,
            std::mutex &mut, std::atomic_bool &should_stop, double delay);