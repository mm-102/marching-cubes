#pragma once
#include <glm/glm.hpp>

template<class T>
class Grid{
    glm::uvec3 m;
    T* data = nullptr;

public:
    T operator()(unsigned x, unsigned y, unsigned z) const {
        return data[z * m.y * m.x + y * m.x + x];
    }
    T& operator()(unsigned x, unsigned y, unsigned z) {
        return data[z * m.y * m.x + y * m.x + x];
    }

    T operator()(glm::uvec3 p) const {
        return data[p.z * m.y * m.x + p.y * m.x + p.x];
    }
    T& operator()(glm::uvec3 p) {
        return data[p.z * m.y * m.x + p.y * m.x + p.x];
    }

    Grid() = default;

    Grid(unsigned mx, unsigned my, unsigned mz) : m(mx,my,mz), data(new T[mx * my * mz]){}
    
    Grid(glm::uvec3 size) : m(size), data(new T[size.x * size.y * size.z]){}

    ~Grid(){
        delete[] data;
    }

    Grid(const Grid& other)
        : m(other.m), data(new T[other.numEle()]) {
        std::copy(other.data, other.data + other.numEle(), data);
    }

    Grid(Grid&& other) noexcept
        : m(other.m), data(other.data) {
        other.data = nullptr;
        other.m = glm::uvec3(0);
    }

    Grid& operator=(const Grid& other) {
        if (this == &other) return *this;
        T* newData = new T[other.numEle()];
        std::copy(other.data, other.data + other.numEle(), newData);
        delete[] data;
        m = other.m;
        data = newData;
        return *this;
    }

    Grid& operator=(Grid&& other) noexcept {
        if (this == &other) return *this;
        delete[] data;
        m = other.m;
        data = other.data;
        other.data = nullptr;
        other.m = glm::uvec3(0);
        return *this;
    }

    glm::uvec3 getSize() const {
        return m;
    }

    size_t numEle() const {
        return m.x * m.y * m.z;
    }

    const T* raw_data() const { return data; }
    T* raw_data() { return data; }
};