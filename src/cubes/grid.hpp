#pragma once
#include <vector>
#include <glm/glm.hpp>

template<class T>
class Grid{
    std::vector<T> data;
    glm::uvec3 m;

public:
    T operator()(unsigned x, unsigned y, unsigned z) const {
        return data[z * m.y * m.x + y * m.x + x];
    }

    T& operator()(unsigned x, unsigned y, unsigned z) {
        return data[z * m.y * m.x + y * m.x + x];
    }

    Grid(unsigned mx, unsigned my, unsigned mz) : m(mx,my,mz), data(mx * my * mz){
        data.resize(mx * my * mz);
    }
    Grid(unsigned mx, unsigned my, unsigned mz, const std::vector<T> &initData) : m(mx,my,mz), data(initData){}

    Grid(glm::uvec3 size) : m(size), data(){
        data.resize(size.x * size.y * size.z);
    }
    Grid(glm::uvec3 size, const std::vector<T> &initData) : m(size), data(initData){}

    glm::uvec3 getSize() const{
        return m;
    }

    const float* vector_data() const { return data.data(); }
    float* vector_data() { return data.data(); }
};