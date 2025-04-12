#include "triangles.hpp"
#include <iostream>

Triangles::Triangles(unsigned max_size) : maxSize(max_size){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO1);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * max_size, 0, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0); 
}

void Triangles::add_verticies(const std::vector<float> &data){
    if(bufferSize + data.size() > maxSize){
        std::cerr << "Triangles: tried to add more data than set max size" << std::endl;
        return;
    }
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);

    glBufferSubData(GL_ARRAY_BUFFER, sizeof(float) * bufferSize, sizeof(float) * data.size(), data.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    bufferSize += data.size();
}

void Triangles::draw(ShaderProgram &shader){
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO1);

    glDrawArrays(GL_TRIANGLES, 0, bufferSize);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}