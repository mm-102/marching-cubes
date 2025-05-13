#include "CudaMC.hpp"
#include "common.hpp"
#include "_CudaMC.hpp"
#include <glm/glm.hpp>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>


namespace CudaMC
{
    template<typename T>
    struct Ele{
        T d;
        float v;
    };

    template<typename T>
    __device__ void grid_cell(const GPU_Grid &grid, uint8_t &idx, const uint &x, const uint &y, const uint &z, T &out);

    template<> inline __device__ void grid_cell<Ele<P>>(const GPU_Grid &grid, uint8_t &idx, const uint &x, const uint &y, const uint &z, Ele<P> &out){
        out.d.x = static_cast<float>(x + d_order[idx][0]);
        out.d.y = static_cast<float>(y + d_order[idx][1]);
        out.d.z = static_cast<float>(z + d_order[idx][2]);

        out.v = grid(
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );
    }

    template<> inline __device__ void grid_cell<Ele<PG>>(const GPU_Grid &grid, uint8_t &idx, const uint &x, const uint &y, const uint &z, Ele<PG> &out){
        
        out.d.p.x = static_cast<float>(x + d_order[idx][0]);
        out.d.p.y = static_cast<float>(y + d_order[idx][1]);
        out.d.p.z = static_cast<float>(z + d_order[idx][2]);

        out.v = grid(
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );

        out.d.g = calcGrad(grid, 
            x + d_order[idx][0],
            y + d_order[idx][1],
            z + d_order[idx][2]
        );
    }

    template<typename T>
    __device__ void interpolate(const Ele<T>& e1, const Ele<T>& e2, float &isovalue, T &out);

    template<> __device__ void  interpolate<P>(const Ele<P>& e1, const Ele<P>& e2, float &isovalue, P &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out = mix(e1.d, e2.d, mu);
    }

    template<> __device__ void interpolate<PG>(const Ele<PG>& e1, const Ele<PG>& e2, float &isovalue, PG &out) {
        float mu = (isovalue - e1.v) / (e2.v - e1.v);
        out.p = mix(e1.d.p, e2.d.p, mu);
        out.g = normalize(mix(e1.d.g, e2.d.g, mu));
    }

    __global__ void count_triangles_kernel(GPU_Grid grid, float isovalue, int *outCounts){
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;
        uint z = blockIdx.z * blockDim.z + threadIdx.z;

        
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

    template<typename T>
    __global__ void triangulate_kernel(GPU_Grid grid, float isovalue, 
            float3* outVerts, float3* outNormals,
            const int *offsets){
        
        uint x = blockIdx.x * blockDim.x + threadIdx.x;
        uint y = blockIdx.y * blockDim.y + threadIdx.y;
        uint z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if(x >= grid.x - 1 || y >= grid.y - 1 || z >= grid.z - 1)
            return;

        float3 p = make_float3(x, y, z);
        
        uint8_t cubeIndex = calcCubeIndex(grid, x, y, z, isovalue);

        if (d_edgeTable[cubeIndex] == 0) return;

        T intersections[12];
        int intersectionsKey = d_edgeTable[cubeIndex];
        Ele<T> ele1, ele2;
        for (int i = 0; i < 12; ++i) {
            if (intersectionsKey & (1 << i)) {
                grid_cell<Ele<T>>(grid, d_edgeToVertices[i][0], x, y, z, ele1);
                grid_cell<Ele<T>>(grid, d_edgeToVertices[i][1], x, y, z, ele2);
                interpolate<T>(ele1, ele2, isovalue, intersections[i]);
            }
        }

        int base = offsets[grid.index_g(p.x,p.y,p.z)] * 3;

        if constexpr (std::is_same_v<T,P>){
            for (int i = 0; d_triTable[cubeIndex][i + 2] != -1; i += 3) {
                const P p1 = intersections[d_triTable[cubeIndex][i]];
                const P p2 = intersections[d_triTable[cubeIndex][i + 1]];
                const P p3 = intersections[d_triTable[cubeIndex][i + 2]];
    
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
                const PG p = intersections[d_triTable[cubeIndex][i]];
                outVerts[base] = p.p;
                outNormals[base] = p.g;
                base++;
            }
        }
    }

    template<typename T>
    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
            std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){
            
        const glm::vec3 size = grid.getSize();
        size_t numEle = size.x * size.y * size.z;
        const int totalCells = (size.x - 1) * (size.y - 1) * (size.z - 1);

        dim3 threads(16,16,4);
        dim3 blocks(
            ceil((double)(size.x - 1) / threads.x),
            ceil((double)(size.y - 1) / threads.y),
            ceil((double)(size.z - 1) / threads.z)
        );

        // std::cout << blocks.x << " " << blocks.y << " " << blocks.z << std::endl;
        // std::cout << blocks.x * threads.x << " " << blocks.y * threads.y << " " << blocks.z * threads.z << std::endl;
        // std::cout << size.x - 1 << " " << size.y - 1 << " " << size.z - 1 << std::endl;

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

        triangulate_kernel<T><<<blocks, threads>>>(d_grid, isovalue, d_verts, d_normals, thrust::raw_pointer_cast(d_offsets.data()));
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

    void setConstMem(){
        cudaMemcpyToSymbol(d_edgeToVertices, edgeToVerticesU, sizeof(edgeToVerticesU));
        cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
        cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
        
        cudaMemcpyToSymbol(d_order, order, sizeof(order));
    }

}

template void CudaMC::trinagulate_grid<CudaMC::P>(const Grid<float> &grid, float isovalue, 
    std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
    
template void CudaMC::trinagulate_grid<CudaMC::PG>(const Grid<float> &grid, float isovalue, 
    std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);