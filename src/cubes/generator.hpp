#pragma once
#include "grid.hpp"
#include <glm/glm.hpp>
#include <string>

namespace gen{

    Grid<float> fromFile(const std::string fileName);
    Grid<float> sphere(float radius);
    Grid<float> torus(float r_minor, float r_major);
    Grid<float> gyroid(glm::uvec3 size, float scale, float thresh = 0.0f);
}