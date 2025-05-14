#pragma once
#include <glm/glm.hpp>

template<class T>
class Grid{
    glm::uvec3 s;
    T iso;
    T* data = nullptr;

public:
    T operator()(unsigned x, unsigned y, unsigned z) const {
        return data[z * s.y * s.x + y * s.x + x];
    }
    T& operator()(unsigned x, unsigned y, unsigned z) {
        return data[z * s.y * s.x + y * s.x + x];
    }

    T operator()(glm::uvec3 p) const {
        return data[p.z * s.y * s.x + p.y * s.x + p.x];
    }
    T& operator()(glm::uvec3 p) {
        return data[p.z * s.y * s.x + p.y * s.x + p.x];
    }

    Grid() = default;

    Grid(unsigned mx, unsigned my, unsigned mz, T isovalue) : s(mx,my,mz), iso(isovalue), data(new T[mx * my * mz]){}
    
    Grid(glm::uvec3 size, T isovalue) : s(size), iso(isovalue), data(new T[size.x * size.y * size.z]){}

    ~Grid(){
        delete[] data;
    }

    Grid(const Grid& other)
        : s(other.s), iso(other.iso), data(new T[other.numEle()]) {
        std::copy(other.data, other.data + other.numEle(), data);
    }

    Grid(Grid&& other) noexcept
        : s(other.s), iso(other.iso), data(other.data) {
        other.data = nullptr;
        other.s = glm::uvec3(0);
    }

    Grid& operator=(const Grid& other) {
        if (this == &other) return *this;
        T* newData = new T[other.numEle()];
        std::copy(other.data, other.data + other.numEle(), newData);
        delete[] data;
        s = other.s;
        iso = other.iso;
        data = newData;
        return *this;
    }

    Grid& operator=(Grid&& other) noexcept {
        if (this == &other) return *this;
        delete[] data;
        s = other.s;
        data = other.data;
        other.data = nullptr;
        other.s = glm::uvec3(0);
        return *this;
    }

    glm::uvec3 getSize() const {
        return s;
    }

    size_t numEle() const {
        return s.x * s.y * s.z;
    }

    T getIsovalue() const {
        return iso;
    }

    const T* raw_data() const { return data; }
    T* raw_data() { return data; }
};