#include "smooth_triangles.hpp"

SmoothTriangles::SmoothTriangles(unsigned max_size, glm::mat4 M) : Triangles(max_size, M){}

void SmoothTriangles::add_verticies(const std::vector<glm::vec3> &data){
    std::vector<glm::vec3> normals(data.size());

    verts.reserve(verts.size() + data.size());
    verts.insert(verts.end(), data.begin(), data.end());

    glm::vec3 p1, p2, p3, norm, cross, cross_inv;
    // for every trinagle
    for(int i = 0; i < data.size() - 2; i += 3){
        p1 = data[i];
        p2 = data[i+1];
        p3 = data[i+2];
        cross = glm::cross(p2-p1, p3-p1);
        norm = glm::normalize(cross);
        normals.at(i) = norm;
        normals.at(i+1) = norm;
        normals.at(i+2) = norm;

        cross_inv = cross * (2.0f / glm::length(cross));

        // if key is not present value is constructed from default glm::vec3 constructor -> (0,0,0)
        vertNormalMap[p1] += cross_inv;
        vertNormalMap[p2] += cross_inv;
        vertNormalMap[p3] += cross_inv;
    }
    Triangles::add_verticies(data, normals);
}

void SmoothTriangles::smooth(){
    std::vector<glm::vec3> normals;
    normals.reserve(verts.size());

    for(auto &vert : verts){
        normals.push_back(glm::normalize(vertNormalMap[vert]));
    }

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO1);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * verts.size(), verts.data());

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3) * normals.size(), normals.data());
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void SmoothTriangles::add_verticies_no_draw(const std::vector<glm::vec3> &data){

    if(bufferSize + data.size() > maxSize){
        std::cerr << "SmoothTriangles: tried to add more data than set max size" << std::endl;
        return;
    }

    verts.reserve(verts.size() + data.size());
    verts.insert(verts.end(), data.begin(), data.end());

    glm::vec3 p1, p2, p3, cross_inv;

    // for every trinagle
    for(int i = 0; i < data.size() - 2; i += 3){
        p1 = data[i];
        p2 = data[i+1];
        p3 = data[i+2];

        cross_inv = glm::cross(p2-p1, p3-p1);

        // if key is not present value is constructed from default glm::vec3 constructor -> (0,0,0)
        vertNormalMap[p1] += cross_inv;
        vertNormalMap[p2] += cross_inv;
        vertNormalMap[p3] += cross_inv;
    }

    bufferSize += data.size();
}