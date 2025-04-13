#pragma once
#include "renderable.hpp"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

class Triangles : public RenderableObject{
    GLuint VAO = 0;
    GLuint VBO1 = 0, VBO2 = 0;

    unsigned bufferSize = 0;
    unsigned maxSize;

public:
    Triangles(unsigned max_size);

    virtual void draw(ShaderProgram &shader);

    // auto calculate normals
    void add_verticies(const std::vector<float> &data);

    void add_verticies(const std::vector<float> &vert_data, const std::vector<float> &norm_data);

    glm::vec4 getModulate() { return glm::vec4(1.0f, 0.3f, 0.2f, 1.0f); };
};