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
    return Grid<float>(sx, sy, sz, std::vector<float>(start,end));
}

Grid<float> Generator::genSphere(glm::vec3 center, float radius){
    Grid<float> sphere(grid_size);

    glm::vec3 to_center;
    float distSquared;
    const float rSquared = radius * radius;

    #pragma omp parallel for private(to_center, distSquared)
    for(int z = 0; z < grid_size.z; z++){
        for(int y = 0; y < grid_size.y; y++){
            for(int x = 0; x < grid_size.x; x++){
                to_center = glm::vec3(x,y,z) - center;
                distSquared = glm::dot(to_center, to_center);
                sphere(x,y,z) = glm::smoothstep(0.0f, 1.0f, distSquared);
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
                const glm::vec3 to_center = glm::vec3(x,z,y) - center;
                const glm::vec2 xy = glm::vec2(to_center);
                float leftSide = r_major - glm::length(xy);
                leftSide *= leftSide;
                leftSide += to_center.z * to_center.z;
                torus(x,y,z) = glm::smoothstep(0.0f, 1.0f, rSquared - leftSide);
            }
        }
    }
    return torus;
}