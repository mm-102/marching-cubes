#pragma once
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include "grid.hpp"
#include <cuda_runtime.h>

__constant__ unsigned d_edgeToVertices[12][2];
__constant__ int d_edgeTable[256];
__constant__ int d_triTable[256][16];

__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return len > 0.0f ? make_float3(v.x / len, v.y / len, v.z / len) : make_float3(0, 0, 0);
}

struct GPU_Grid {
    float* data;
    const unsigned x, y, z;
    const unsigned s;
    const unsigned ne;

    __host__ __device__
    GPU_Grid(float* data, unsigned x, unsigned y, unsigned z)
        : data(data), x(x), y(y), z(z), s(x*y*z), ne((x-1)*(y-1)*(z-1)) {}

    __host__ __device__
    float operator()(unsigned i, unsigned j, unsigned k) const {
        return data[k * y * x + j * x + i];
    }

    __host__ __device__
    float& operator()(unsigned i, unsigned j, unsigned k) {
        return data[k * y * x + j * x + i];
    }

    __host__ __device__
    unsigned index(unsigned i, unsigned j, unsigned k) const {
        return k * y * x + j * x + i;
    }

    __host__ __device__
    unsigned index_g(unsigned i, unsigned j, unsigned k) const {
        return k * (y - 1) * (x - 1) + j * (x - 1) + i;
    }

    __host__ __device__
    unsigned size() const {
        return s;
    }

    __host__ __device__
    unsigned numEle() const {
        return ne;
    }
};

struct FlatGridCellEle {
    float3 p;
    float v;
};

struct FlatGridCell {
    FlatGridCellEle data[8];
};

namespace CudaMarchingCubes
{
    void trinagulate_grid_flat(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);

    void trinagulate_grid(const Grid<float> &grid, float isovalue, 
        std::vector<glm::vec3>& outVerts, std::vector<glm::vec3>& outNormals);
}