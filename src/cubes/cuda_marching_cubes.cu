#include "marching_cubes_common.hpp"
#include "cuda_marching_cubes.hpp"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

__device__ inline int calcCubeIndex(const FlatGridCell& cell, float isovalue) {
    int index = 0;
    for (int i = 0; i < 8; i++) {
        if (cell.data[i].v < isovalue) index |= (1 << i);
    }
    return index;
}

__device__ inline float3 interpolate(const FlatGridCellEle& e1, const FlatGridCellEle& e2, float isovalue) {
    float mu = (isovalue - e1.v) / (e2.v - e1.v);
    return make_float3(
        e1.p.x + mu * (e2.p.x - e1.p.x),
        e1.p.y + mu * (e2.p.y - e1.p.y),
        e1.p.z + mu * (e2.p.z - e1.p.z)
    );
}

__device__ inline void intersection_coords(
    const FlatGridCell& cell, float isovalue, int cubeIndex, float3 intersections[12]
) {
    int intersectionsKey = d_edgeTable[cubeIndex];

    for (int i = 0; i < 12; ++i) {
        if (intersectionsKey & (1 << i)) {
            const unsigned v1 = d_edgeToVertices[i][0];
            const unsigned v2 = d_edgeToVertices[i][1];
            intersections[i] = interpolate(cell.data[v1], cell.data[v2], isovalue);
        }
    }
}

__device__ inline void triangulate_cell_gpu(
    const FlatGridCell& cell, float isovalue,
    float3* outVerts, float3* outNormals, const int *offsets, int tid
) {
    int cubeIndex = calcCubeIndex(cell, isovalue);

    if (d_edgeTable[cubeIndex] == 0) return;
    float3 intersections[12];
    intersection_coords(cell, isovalue, cubeIndex, intersections);

    int base = offsets[tid] * 3;
    for (int i = 0; d_triTable[cubeIndex][i + 2] != -1; i += 3) {
        float3 p1 = intersections[d_triTable[cubeIndex][i]];
        float3 p2 = intersections[d_triTable[cubeIndex][i + 1]];
        float3 p3 = intersections[d_triTable[cubeIndex][i + 2]];

        float3 u = make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        float3 v = make_float3(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);

        float3 norm = normalize(cross(u, v));

        outVerts[base]     = p1;
        outVerts[base + 1] = p2;
        outVerts[base + 2] = p3;

        outNormals[base]     = norm;
        outNormals[base + 1] = norm;
        outNormals[base + 2] = norm;
        base += 3;
    }
}

__global__ void triangulate_flat_kernel(GPU_Grid grid, float isovalue, 
        float3* outVerts, float3* outNormals,
        const int *offsets){
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(x >= grid.x - 1 || y >= grid.y - 1 || z >= grid.z - 1)
        return;

    float3 p = make_float3(x, y, z);

    FlatGridCell cell = {
        FlatGridCellEle{make_float3(p.x,     p.y,     p.z    ), grid(x,   y,   z  )},
        FlatGridCellEle{make_float3(p.x+1.f, p.y,     p.z    ), grid(x+1, y,   z  )},
        FlatGridCellEle{make_float3(p.x+1.f, p.y,     p.z+1.f), grid(x+1, y,   z+1)},
        FlatGridCellEle{make_float3(p.x,     p.y,     p.z+1.f), grid(x,   y,   z+1)},
        FlatGridCellEle{make_float3(p.x,     p.y+1.f, p.z    ), grid(x,   y+1, z  )},
        FlatGridCellEle{make_float3(p.x+1.f, p.y+1.f, p.z    ), grid(x+1, y+1, z  )},
        FlatGridCellEle{make_float3(p.x+1.f, p.y+1.f, p.z+1.f), grid(x+1, y+1, z+1)},
        FlatGridCellEle{make_float3(p.x,     p.y+1.f, p.z+1.f), grid(x,   y+1, z+1)}
    };

    triangulate_cell_gpu(cell, isovalue, outVerts, outNormals, offsets, grid.index_g(p.x,p.y,p.z));
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

namespace CudaMarchingCubes
{
    void trinagulate_grid_flat(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        cudaMemcpyToSymbol(d_edgeToVertices, edgeToVerticesU, sizeof(edgeToVerticesU));
        cudaMemcpyToSymbol(d_edgeTable, edgeTable, sizeof(edgeTable));
        cudaMemcpyToSymbol(d_triTable, triTable, sizeof(triTable));
            
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


        triangulate_flat_kernel<<<blocks, threads>>>(d_grid, isovalue, d_verts, d_normals, thrust::raw_pointer_cast(d_offsets.data()));
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
        // cudaDeviceReset();
    }

    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals){

        }
}