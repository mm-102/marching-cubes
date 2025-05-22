#include "generator.hpp"
#include <fstream>
#include <iterator>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#define STB_PERLIN_IMPLEMENTATION
#include "stb_perlin.h"

namespace gen{

    Grid<float> fromFile(const std::string fileName){
        std::ifstream file(fileName);
        if(!file.is_open()){
            std::cerr << "Generator: Error opening file: " << fileName << std::endl;
        }
        
        unsigned sx, sy, sz;
        float iso;
        file >> sx >> sy >> sz >> iso;

        std::istream_iterator<float> start(file), end;
        Grid<float> grid(sx,sy,sz,iso);
        std::copy(start, end, grid.raw_data());
        return grid;
    }

    Grid<float> sphere(float radius){
        unsigned d = std::ceil(2.0f * radius) + 4;
        glm::vec3 center = glm::vec3(d) * 0.5f;

        Grid<float> sphere(glm::uvec3(d), 0.0f);

        #pragma omp parallel for
        for(unsigned z = 0; z < d; z++){
            for(unsigned y = 0; y < d; y++){
                for(unsigned x = 0; x < d; x++){
                    sphere(x,y,z) = radius - glm::length(glm::vec3(x,y,z) - center);
                }
            }
        }
        
        return sphere;
    }

    Grid<float> torus(float r_minor, float r_major){

        unsigned dxz = static_cast<unsigned>(std::ceil(2.0f * (r_minor + r_major)) + 4);
        unsigned dy = static_cast<unsigned>(std::ceil(2.0f * r_minor) + 4);

        glm::vec3 center = glm::vec3(dxz,dy,dxz) * 0.5f;

        Grid<float> torus(dxz,dy,dxz, 0.0f);

        float rSquared = r_minor * r_minor;
        #pragma omp parallel for schedule(static)
        for(unsigned z = 0; z < dxz; z++){
            for(unsigned y = 0; y < dy; y++){
                #pragma omp simd
                for(unsigned x = 0; x < dxz; x++){
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

    Grid<float> gyroid(glm::uvec3 size, float scale, float thresh){
        Grid<float> field(size, 0.0f);

        #pragma omp parallel for
        for (int z = 0; z < size.z; z++) {
            for (int y = 0; y < size.y; y++) {
                for (int x = 0; x < size.x; x++) {
                    glm::vec3 p = glm::vec3(x, y, z) * scale;
                    float v = std::sin(p.x) * std::cos(p.y) + std::sin(p.y) * std::cos(p.z) + std::sin(p.z) * std::cos(p.x);
                    field(x, y, z) = v - thresh;  // isosurface at 0
                }
            }
        }

        return field;
    }

    Grid<float> perlin(glm::uvec3 size, float scale, glm::uvec3 wrap){
        Grid<float> field(size, 0.0f);

        glm::vec3 mul = glm::vec3(scale) / glm::vec3(size);

        #pragma omp parallel for
        for (int z = 0; z < size.z; z++) {
            for (int y = 0; y < size.y; y++) {
                for (int x = 0; x < size.x; x++) {
                   
                    glm::vec3 c = glm::vec3(x,y,z) * mul;
                    field(x,y,z) = stb_perlin_noise3(c.x, c.y, c.z, wrap.x, wrap.y, wrap.z);
                }
            }
        }

        return field;
    }
}