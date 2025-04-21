#include "triangles.hpp"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

Triangles::Triangles(size_t init_size, glm::mat4 M) : bufferOff(0), bufferSize(init_size), M(M){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &VBO2);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * init_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * init_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    verts.reserve(init_size);
    norms.reserve(init_size);
}

void Triangles::add_verticies(const std::vector<glm::vec3> &vert_data, const std::vector<glm::vec3> &norm_data){
    if(vert_data.size() != norm_data.size()){
        std::cerr << "Triangles: norm and vert counts do not match" << std::endl;
        return;
    }

    if(verts.capacity() < verts.size() + vert_data.size()){
        size_t res = std::max(verts.size() + vert_data.size(), static_cast<size_t>(1.5f * verts.size()));
        verts.reserve(res);
        norms.reserve(res);
    }
    verts.insert(verts.end(), vert_data.begin(), vert_data.end());
    norms.insert(norms.end(), norm_data.begin(), norm_data.end());

    glBindVertexArray(VAO);

    if(bufferOff + vert_data.size() > bufferSize){
        bufferSize = std::max(bufferOff + vert_data.size(), static_cast<size_t>(1.5f * bufferSize));
        
        glBindBuffer(GL_ARRAY_BUFFER, VBO1);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferSize, 0, GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * verts.size(), verts.data());

        glBindBuffer(GL_ARRAY_BUFFER, VBO2);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferSize, 0, GL_DYNAMIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * norms.size(), norms.data());

        bufferOff = verts.size();
    }

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferOff, sizeof(glm::vec3) * vert_data.size(), vert_data.data());

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferOff, sizeof(glm::vec3) * norm_data.size(), norm_data.data()); 

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    bufferOff += vert_data.size();
}

void Triangles::add_verticies(const std::vector<glm::vec3> &data){
    std::vector<glm::vec3> normals(data.size());

    glm::vec3 p1, p2, p3, norm;
    // for every trinagle
    for(int i = 0; i < data.size() - 2; i += 3){
        p1 = data[i];
        p2 = data[i+1];
        p3 = data[i+2];
        norm = glm::normalize(glm::cross(p2-p1, p3-p1));
        normals.at(i) = norm;
        normals.at(i+1) = norm;
        normals.at(i+2) = norm;
    }

    add_verticies(data, normals);
}

void Triangles::draw(ShaderProgram &shader){
    const static glm::vec4 inside_modulate(0.2f, 0.3f, 1.0f, 1.0f);
    glUniform4fv(shader.u("uInModulate"), 1, glm::value_ptr(inside_modulate));
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, bufferSize);
    glBindVertexArray(0);
}