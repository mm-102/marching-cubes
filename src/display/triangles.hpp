#pragma once
#include "renderable.hpp"
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <vector>

class Triangles : public RenderableObject{
protected:
    GLuint VAO = 0;
    GLuint VBO1 = 0, VBO2 = 0;

    size_t bufferOff;
    size_t bufferSize;
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> norms;

    glm::mat4 M;

public:
    Triangles(size_t init_size, glm::mat4 M = glm::mat4(1.0f));

    virtual void draw(ShaderProgram &shader);

    // auto calculate normals
    void add_verticies(const std::vector<glm::vec3> &data);

    void add_verticies(const std::vector<glm::vec3> &vert_data, const std::vector<glm::vec3> &norm_data);

    glm::vec4 getModulate() { return glm::vec4(1.0f, 0.3f, 0.2f, 1.0f); };
    glm::mat4 getModelMatrix() { return M; };
};