#include "triangles.hpp"
#include <iostream>

Triangles::Triangles(unsigned max_size) : maxSize(max_size){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);
    glGenBuffers(1, &VBO2);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * max_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * max_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0); 
}

void Triangles::add_verticies(const std::vector<float> &vert_data, const std::vector<float> &norm_data){
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
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * bufferSize, sizeof(float) * vert_data.size(), vert_data.data());

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * bufferSize, sizeof(float) * norm_data.size(), norm_data.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    bufferSize += vert_data.size();
}

void Triangles::add_verticies(const std::vector<float> &data){
    std::vector<float> normals(data.size());

    glm::vec3 p1, p2, p3, norm;
    // for every trinagle
    for(int i = 0; i < data.size() - 8; i += 9){
        p1 = glm::vec3(data[i], data[i+1], data[i+2]);
        p2 = glm::vec3(data[i+3], data[i+4], data[i+5]);
        p3 = glm::vec3(data[i+6], data[i+7], data[i+8]);
        norm = glm::normalize(glm::cross(p2-p1, p3-p1));
        for(int j = i; j < i + 9; j += 3){
            normals[j] = norm.x;
            normals[j+1] = norm.y;
            normals[j+2] = norm.z;
        }
    }

    for(auto v : data){
        std::cout << v << " ";
    }
    std::cout << std::endl;
    for(auto v : normals){
        std::cout << v << " ";
    }
    std::cout << std::endl;

    add_verticies(data, normals);
}

void Triangles::draw(ShaderProgram &shader){
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, bufferSize);
    glBindVertexArray(0);
}