#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <glm/glm.hpp>

void saveOBJ(const std::string &filename, 
        const std::vector<glm::vec3> &verts, 
        const std::vector<glm::vec3> &norms){
    
    if(verts.size() != norms.size()){
        std::cerr << "saveOBJ: verts and norms size mismatch: " << verts.size() << " " << norms.size() << std::endl;
        return;
    }

    if(verts.size() % 3){
        std::cerr << "saveOBJ: incorrect vector size (expected triangles): " << verts.size() << std::endl;
        return;
    }

    std::ofstream file(filename);

    if(!file){
        std::cerr << "saveOBJ: error opening file: " << filename << std::endl;
        return;
    }

    for(const auto &v : verts){
        file << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }

    for(const auto &n : norms){
        file << "vn " << n.x << " " << n.y << " " << n.z << "\n";
    }

    for (size_t i = 0; i < verts.size(); i += 3) {

        file << "f ";
        for (size_t j = 0; j < 3; ++j) {
            size_t idx = i + j + 1;
            file << idx << "//" << idx << " ";
        }
        file << "\n";
    }

    file.close();

    std::cout << "Mesh saved to: " << filename << std::endl;
}