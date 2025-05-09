#include "marching_cubes_common.hpp"
#include "cuda_marching_cubes.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


namespace CudaMC
{
    template<typename Traits>
    __device__ inline uint8_t calcCubeIndex(const typename Traits::GridCell &cell, float isovalue) {
        uint8_t index = 0;
        // for (int i = 0; i < 8; i++) {
        //     if (cell.data[i].v < isovalue) index |= (1 << i);
        // }
        if(cell.data[0].v < isovalue) index |= 1;
        if(cell.data[1].v < isovalue) index |= 2;
        if(cell.data[2].v < isovalue) index |= 4;
        if(cell.data[3].v < isovalue) index |= 8;
        if(cell.data[4].v < isovalue) index |= 16;
        if(cell.data[5].v < isovalue) index |= 32;
        if(cell.data[6].v < isovalue) index |= 64;
        if(cell.data[7].v < isovalue) index |= 128;
        return index;
    }

    __device__ uint8_t calcCubeIndex(const GPU_Grid &grid, const int &x, const int &y, const int &z, const float &isovalue){
        int cubeIndex = 0;
        if (grid(x,   y,   z  ) < isovalue) cubeIndex |= 1;
        if (grid(x+1, y,   z  ) < isovalue) cubeIndex |= 2;
        if (grid(x+1, y,   z+1) < isovalue) cubeIndex |= 4;
        if (grid(x,   y,   z+1) < isovalue) cubeIndex |= 8;
        if (grid(x,   y+1, z  ) < isovalue) cubeIndex |= 16;
        if (grid(x+1, y+1, z  ) < isovalue) cubeIndex |= 32;
        if (grid(x+1, y+1, z+1) < isovalue) cubeIndex |= 64;
        if (grid(x,   y+1, z+1) < isovalue) cubeIndex |= 128;
        return cubeIndex;
    }

    __device__ float3 calcGrad(const GPU_Grid &grid, const int &x, const int &y, const int &z){   
        if(x < 1 || y < 1 || z < 1)
            return make_float3(0,0,0);
             
        return normalize(make_float3(
            grid(x + 1, y, z) - grid(x - 1, y, z),
            grid(x, y + 1, z) - grid(x, y - 1, z),
            grid(x, y, z + 1) - grid(x, y, z - 1)
        ));
    }

    struct Flat{
        using GridCell = struct {
            struct Ele{
                float3 p;
                float v;
            } data[8];
        };

        using inter_t = float3;

        struct MakeCell {
            __device__
            GridCell operator()(const GPU_Grid &grid, float3 &p, const int &x, const int &y, const int &z){
                return GridCell{ .data {
                    {make_float3(x,     y,     z    ), grid(x,   y,   z  )},
                    {make_float3(x+1.f, y,     z    ), grid(x+1, y,   z  )},
                    {make_float3(x+1.f, y,     z+1.f), grid(x+1, y,   z+1)},
                    {make_float3(x,     y,     z+1.f), grid(x,   y,   z+1)},
                    {make_float3(x,     y+1.f, z    ), grid(x,   y+1, z  )},
                    {make_float3(x+1.f, y+1.f, z    ), grid(x+1, y+1, z  )},
                    {make_float3(x+1.f, y+1.f, z+1.f), grid(x+1, y+1, z+1)},
                    {make_float3(x,     y+1.f, z+1.f), grid(x,   y+1, z+1)}
                }};
            }
        };
    };

    struct Grad{
        using GridCell = struct {
            struct Ele{
                float3 p;
                float v;
                float3 g;
            } data[8];
        };

        using inter_t = struct {
            float3 p;
            float3 g;
        };

        struct MakeCell {
            __device__
            GridCell operator()(const GPU_Grid &grid, float3 &p, const int &x, const int &y, const int &z){
                return GridCell{ .data {
                    {make_float3(p.x,     p.y,     p.z    ), grid(x,   y,   z  ), calcGrad(grid, x,   y,   z  )},
                    {make_float3(p.x+1.f, p.y,     p.z    ), grid(x+1, y,   z  ), calcGrad(grid, x+1, y,   z  )},
                    {make_float3(p.x+1.f, p.y,     p.z+1.f), grid(x+1, y,   z+1), calcGrad(grid, x+1, y,   z+1)},
                    {make_float3(p.x,     p.y,     p.z+1.f), grid(x,   y,   z+1), calcGrad(grid, x,   y,   z+1)},
                    {make_float3(p.x,     p.y+1.f, p.z    ), grid(x,   y+1, z  ), calcGrad(grid, x,   y+1, z  )},
                    {make_float3(p.x+1.f, p.y+1.f, p.z    ), grid(x+1, y+1, z  ), calcGrad(grid, x+1, y+1, z  )},
                    {make_float3(p.x+1.f, p.y+1.f, p.z+1.f), grid(x+1, y+1, z+1), calcGrad(grid, x+1, y+1, z+1)},
                    {make_float3(p.x,     p.y+1.f, p.z+1.f), grid(x,   y+1, z+1), calcGrad(grid, x,   y+1, z+1)}
                }};
            }
        };
    };

    template<typename Traits>
    __device__ void grid_cell(const GPU_Grid &grid, unsigned &idx, const int &x, const int &y, const int &z, typename Traits::GridCell::Ele &out);

    template<> inline __device__ void grid_cell<Flat>(const GPU_Grid &grid, unsigned &idx, const int &x, const int &y, const int &z, typename Flat::GridCell::Ele &out){
        out.p.x = static_cast<float>(x + d_order[idx][0]);
        out.p.y = static_cast<float>(y + d_order[idx][1]);
        out.p.z = static_cast<float>(z + d_order[idx][2]);

        out.v = grid(
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );
    }

    template<> inline __device__ void grid_cell<Grad>(const GPU_Grid &grid, unsigned &idx, const int &x, const int &y, const int &z, typename Grad::GridCell::Ele &out){
        
        out.p.x = static_cast<float>(x + d_order[idx][0]);
        out.p.y = static_cast<float>(y + d_order[idx][1]);
        out.p.z = static_cast<float>(z + d_order[idx][2]);

        out.v = grid(
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );

        out.g = calcGrad(grid, 
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );
    }

    __global__ void count_triangles_kernel(GPU_Grid grid, float isovalue, int *outCounts){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

        
        if(x >= grid.x - 1 || y >= grid.y - 1 || z >= grid.z - 1)
            return;
        
        int cubeIndex = 0;
        if (grid(x,   y,   z  ) < isovalue) cubeIndex |= 1;
        if (grid(x+1, y,   z  ) < isovalue) cubeIndex |= 2;
        if (grid(x+1, y,   z+1) < isovalue) cubeIndex |= 4;
        if (grid(x,   y,   z+1) < isovalue) cubeIndex |= 8;
        if (grid(x,   y+1, z  ) < isovalue) cubeIndex |= 16;
        if (grid(x+1, y+1, z  ) < isovalue) cubeIndex |= 32;
        if (grid(x+1, y+1, z+1) < isovalue) cubeIndex |= 64;
        if (grid(x,   y+1, z+1) < isovalue) cubeIndex |= 128;

        int count = 0;
        for (int i = 0; d_triTable[cubeIndex][i] != -1; i += 3)
            ++count;

        outCounts[grid.index_g(x,y,z)] = count;
    }

    template<typename Traits>
    __device__ void interpolate(const typename Traits::GridCell::Ele& e1, const typename Traits::GridCell::Ele& e2, float &isovalue, typename Traits::inter_t &out);

    template<> __device__ void  interpolate<Flat>(const typename Flat::GridCell::Ele& e1, const typename Flat::GridCell::Ele& e2, float &isovalue, typename Flat::inter_t &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out = mix(e1.p, e2.p, mu);
    }

    template<> __device__ void interpolate<Grad>(const typename Grad::GridCell::Ele& e1, const typename Grad::GridCell::Ele& e2, float &isovalue, typename Grad::inter_t &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out.p = mix(e1.p, e2.p, mu);
        out.g = normalize(mix(e1.g, e2.g, mu));
    }

    template<typename Traits>
    __device__ inline void intersection_coords(
        const typename Traits::GridCell& cell, float &isovalue, const int &cubeIndex, 
        typename Traits::inter_t intersections[12]
    ) {
        int intersectionsKey = d_edgeTable[cubeIndex];

        for (int i = 0; i < 12; ++i) {
            if (intersectionsKey & (1 << i)) {
                interpolate<Traits>(cell.data[d_edgeToVertices[i][0]], cell.data[d_edgeToVertices[i][1]], isovalue, intersections[i]);
            }
        }


    }

    template<typename Traits>
    __global__ void triangulate_kernel(GPU_Grid grid, float isovalue, 
            float3* outVerts, float3* outNormals,
            const int *offsets){
        
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(x >= grid.x - 1 || y >= grid.y - 1 || z >= grid.z - 1)
            return;

        float3 p = make_float3(x, y, z);
        
        // typename Traits::MakeCell make_cell;
        // typename Traits::GridCell cell = make_cell(grid, p, x, y, z);

        // uint8_t cubeIndex = calcCubeIndex<Traits>(cell, isovalue);
        uint8_t cubeIndex = calcCubeIndex(grid, x, y, z, isovalue);

        if (d_edgeTable[cubeIndex] == 0) return;

        typename Traits::inter_t intersections[12];
        // intersection_coords<Traits>(cell, isovalue, cubeIndex, intersections);
        int intersectionsKey = d_edgeTable[cubeIndex];
        typename Traits::GridCell::Ele ele1, ele2;
        for (int i = 0; i < 12; ++i) {
            if (intersectionsKey & (1 << i)) {
                grid_cell<Traits>(grid, d_edgeToVertices[i][0], x, y, z, ele1);
                grid_cell<Traits>(grid, d_edgeToVertices[i][1], x, y, z, ele2);
                interpolate<Traits>(ele1, ele2, isovalue, intersections[i]);
            }
        }


        int base = offsets[grid.index_g(p.x,p.y,p.z)] * 3;

        if constexpr (std::is_same_v<Traits,Flat>){
            for (int i = 0; d_triTable[cubeIndex][i + 2] != -1; i += 3) {
                const auto p1 = intersections[d_triTable[cubeIndex][i]];
                const auto p2 = intersections[d_triTable[cubeIndex][i + 1]];
                const auto p3 = intersections[d_triTable[cubeIndex][i + 2]];
    
                const float3 u = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
                const float3 v = make_float3(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);
    
                const float3 norm = normalize(cross(u, v));
    
                outVerts[base]     = p1;
                outVerts[base + 1] = p2;
                outVerts[base + 2] = p3;
    
                outNormals[base]     = norm;
                outNormals[base + 1] = norm;
                outNormals[base + 2] = norm;
                base += 3;
            }
        } else {
            for(int i = 0; d_triTable[cubeIndex][i] != -1; i++){
                const auto p = intersections[d_triTable[cubeIndex][i]];
                outVerts[base] = p.p;
                outNormals[base] = p.g;
                base++;
            }
        }
    }

    template<typename Traits>
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){
        
        cudaMemcpyToSymbol(d_edgeToVertices, edgeToVerticesU, sizeof(edgeToVerticesU));
        cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
        cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
        
        cudaMemcpyToSymbol(d_order, order, sizeof(order));
            
        const glm::vec3 size = grid.getSize();
        size_t numEle = size.x * size.y * size.z;
        const int totalCells = (size.x - 1) * (size.y - 1) * (size.z - 1);

        dim3 threads(16,16,4);
        dim3 blocks(
            ceil((double)(size.x - 1) / threads.x),
            ceil((double)(size.y - 1) / threads.y),
            ceil((double)(size.z - 1) / threads.z)
        );

        std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
        std::cout << blocks.x * threads.x << " " << blocks.y * threads.y << " " << blocks.z * threads.z << std::endl;
        std::cout << size.x - 1 << " " << size.y - 1 << " " << size.z - 1 << std::endl;

        float *d_data;
        cudaMalloc(&d_data, numEle * sizeof(float));
        cudaMemcpy(d_data, grid.vector_data(), numEle * sizeof(float), cudaMemcpyHostToDevice);

        GPU_Grid d_grid(d_data, size.x, size.y, size.z);

        thrust::device_vector<int> d_counts(totalCells + 1);
        thrust::device_vector<int> d_offsets(totalCells + 1);

        count_triangles_kernel<<<blocks, threads>>>(d_grid, isovalue, thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[count_triangles] CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        // prefix sum
        thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[exclusive_scan] CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        int triNum;
        cudaMemcpy(&triNum, thrust::raw_pointer_cast(d_offsets.data()+totalCells), sizeof(int), cudaMemcpyDeviceToHost);
        const int vertNum = 3 * triNum;

        float3 *d_verts;
        float3* d_normals;
        cudaMalloc(&d_verts, vertNum * sizeof(float3));
        cudaMalloc(&d_normals, vertNum * sizeof(float3));


        triangulate_kernel<Traits><<<blocks, threads>>>(d_grid, isovalue, d_verts, d_normals, thrust::raw_pointer_cast(d_offsets.data()));
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[triangulate] CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        outVerts.resize(vertNum);
        outNormals.resize(vertNum);

        cudaMemcpy(outVerts.data(), d_verts, vertNum * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(outNormals.data(), d_normals, vertNum * sizeof(float3), cudaMemcpyDeviceToHost);

        cudaFree(d_data);
        cudaFree(d_verts);
        cudaFree(d_normals);
    }
}
