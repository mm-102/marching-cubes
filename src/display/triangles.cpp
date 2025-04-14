#include "triangles.hpp"
#include <iostream>

Triangles::Triangles(unsigned max_size, glm::mat4 M) : maxSize(max_size), M(M){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &VBO2);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * max_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * max_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0); 
}

void Triangles::add_verticies(const std::vector<glm::vec3> &vert_data, const std::vector<glm::vec3> &norm_data){
    if(vert_data.size() != norm_data.size()){
        std::cerr << "Triangles: norm and vert counts do not match" << std::endl;
        return;
    }
    if(bufferSize + vert_data.size() > maxSize){
        std::cerr << "Triangles: tried to add more data than set max size" << std::endl;
        return;
    }

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferSize, sizeof(glm::vec3) * vert_data.size(), vert_data.data());

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * bufferSize, sizeof(glm::vec3) * norm_data.size(), norm_data.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    bufferSize += vert_data.size();
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
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, bufferSize);
    glBindVertexArray(0);
}