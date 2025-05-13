#include "generator.hpp"
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

Generator::Generator(glm::uvec3 grid_size) : grid_size(grid_size){}
Generator::Generator(unsigned sx, unsigned sy, unsigned sz) : grid_size(sx, sy, sz){}

Grid<float> Generator::fromFile(const std::string fileName){
    std::ifstream file(fileName);
    if(!file.is_open()){
        std::cerr << "Generator: Error opening file: " << fileName << std::endl;
    }
    
    unsigned sx, sy, sz;
    file >> sx >> sy >> sz;

    std::istream_iterator<float> start(file), end;
    Grid<float> grid(sx,sy,sz);
    std::copy(start, end, grid.raw_data());
    return grid;
}

Grid<float> Generator::genSphere(glm::vec3 center, float radius){
    Grid<float> sphere(grid_size);

    #pragma omp parallel for
    for(int z = 0; z < grid_size.z; z++){
        for(int y = 0; y < grid_size.y; y++){
            for(int x = 0; x < grid_size.x; x++){
                sphere(x,y,z) = radius - glm::length(glm::vec3(x,y,z) - center);
            }
        }
    }
    
    return sphere;
}

Grid<float> Generator::genTorus(glm::vec3 center, float r_minor, float r_major){
    Grid<float> torus(grid_size);

    float rSquared = r_minor * r_minor;
    #pragma omp parallel for schedule(static)
    for(int z = 0; z < grid_size.z; z++){
        for(int y = 0; y < grid_size.y; y++){
            #pragma omp simd
            for(int x = 0; x < grid_size.x; x++){
                const glm::vec3 to_center = glm::vec3(x,y,z) - center;
                const glm::vec2 xz = glm::vec2(to_center.x, to_center.z);
                float leftSide = r_major - glm::length(xz);
                leftSide *= leftSide;
                leftSide += to_center.y * to_center.y;
                torus(x,y,z) = rSquared - leftSide;
            }
        }
    }
    return torus;
}

Grid<float> Generator::genGyroid(float scale, float threshold) {
    Grid<float> field(grid_size);

    #pragma omp parallel for
    for (int z = 0; z < grid_size.z; z++) {
        for (int y = 0; y < grid_size.y; y++) {
            for (int x = 0; x < grid_size.x; x++) {
                glm::vec3 p = glm::vec3(x, y, z) * scale;
                float v = std::sin(p.x) * std::cos(p.y) + std::sin(p.y) * std::cos(p.z) + std::sin(p.z) * std::cos(p.x);
                field(x, y, z) = v - threshold;  // isosurface at 0
            }
        }
    }

    return field;
}