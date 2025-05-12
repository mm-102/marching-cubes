#pragma once
#include <glm/glm.hpp>
#include <cstdint>
#include <cuda_runtime.h>

__constant__ uint8_t d_edgeToVertices[12][2];
__constant__ uint16_t d_edgeTable[256];
__constant__ int8_t d_triTable[256][16];

__constant__ uint8_t d_order[8][3];

const uint8_t order[8][3]{
    {0,0,0},
    {1,0,0},
    {1,0,1},
    {0,0,1},
    {0,1,0},
    {1,1,0},
    {1,1,1},
    {0,1,1},
};

__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float3 mix(float3 e1, float3 e2, float v) {
    return make_float3(
        e1.x + v * (e2.x - e1.x),
        e1.y + v * (e2.y - e1.y),
        e1.z + v * (e2.z - e1.z)
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

__device__ float3 calcGrad(const GPU_Grid &grid, const int &x, const int &y, const int &z){   
    if(x < 1 || y < 1 || z < 1)
        return make_float3(0,0,0);
            
    return normalize(make_float3(
        grid(x + 1, y, z) - grid(x - 1, y, z),
        grid(x, y + 1, z) - grid(x, y - 1, z),
        grid(x, y, z + 1) - grid(x, y, z - 1)
    ));
}

__device__ uint8_t calcCubeIndex(const GPU_Grid &grid, const int &x, const int &y, const int &z, const float &isovalue){
    uint8_t cubeIndex = 0;
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