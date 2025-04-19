#pragma once
#include "triangles.hpp"
#include "unordered_map"

class SmoothTriangles: public Triangles{

    struct Vec3ApproxEqual {
        bool operator()(const glm::vec3& a, const glm::vec3& b) const {
            const float epsilon = 1e-7f;
            return glm::all(glm::lessThan(glm::abs(a - b), glm::vec3(epsilon)));
        }
    };

    struct Vec3ApproxHasher {
        std::size_t operator()(const glm::vec3& v) const {
            const static float gridSize = 1e6f;
            const glm::ivec3 vg(v * gridSize);
    
            std::size_t hx = std::hash<int>{}(vg.x);
            std::size_t hy = std::hash<int>{}(vg.y);
            std::size_t hz = std::hash<int>{}(vg.z);
    
            return ((hx ^ (hy << 1)) >> 1) ^ (hz << 1);
        }
    };

    std::unordered_map<glm::vec3,glm::vec3,Vec3ApproxHasher,Vec3ApproxEqual> vertNormalMap;
    std::vector<glm::vec3> verts;

public:
    SmoothTriangles(unsigned max_size, glm::mat4 M);

    // auto calculate normals
    void add_verticies(const std::vector<glm::vec3> &data);

    void add_verticies_no_draw(const std::vector<glm::vec3> &data);

    void smooth();
};