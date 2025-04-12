#pragma once
#include "renderable.hpp"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

class Triangles : public RenderableObject{
    GLuint VAO = 0;
    GLuint VBO1 = 0;

    unsigned bufferSize = 0;
    unsigned maxSize;

public:
    Triangles(unsigned max_size);

    virtual void draw(ShaderProgram &shader);
    void add_verticies(const std::vector<float> &data);
};