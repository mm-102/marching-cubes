#pragma once
#include "grid.hpp"
#include <glm/glm.hpp>
#include <string>

class Generator{
    glm::uvec3 grid_size;

public: 
    Generator(glm::uvec3 grid_size);
    Generator(unsigned sx, unsigned sy, unsigned sz);

    static Grid<float> fromFile(const std::string fileName);
    Grid<float> genSphere(glm::vec3 center, float radius);
    Grid<float> genTorus(glm::vec3 center, float r_minor, float r_major);
    Grid<float> genGyroid(float scale, float threshold = 0.0f);
};