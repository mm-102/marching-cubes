#include "cxxopts.hpp"
#include <vector>
#include <string>
#include <generator.hpp>
#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <grid.hpp>
#include <CudaMC.hpp>
#include <CpuMC.hpp>
#include <cstdint>
#include <thread>

int main(int argc, const char *argv[]){

    unsigned dim;
    std::string shape;

    cxxopts::Options options("Marching cube testing", "measure time");

    options.add_options()
        ("d,dim","dim", cxxopts::value<unsigned>(dim)->default_value("100"))
        ("s,shape","shape", cxxopts::value<std::string>(shape)->default_value("perlin"))
    ;

    auto result = options.parse(argc, argv);

    CudaMC::setConstMem();

    int64_t seq_res = 0, omp_res = 0, cuda_res_ker = 0, cuda_res_cop = 0;
    volatile size_t c_seq = 0, c_omp = 0, c_cuda = 0;

    Grid<float> grid;
    
    if(shape == "perlin")
        grid = gen::perlin(glm::uvec3(dim), static_cast<float>(dim) / 10.0f);
    else if (shape == "sphere")
        grid = gen::sphere(static_cast<float>(dim) * 0.5 - 4);
    else{
        std::cerr << "Invalid shape!" << std::endl;
        exit(EXIT_FAILURE);
    }

    {
        std::vector<glm::vec3> vertData, normData;
        auto start = std::chrono::steady_clock::now();
        CpuMC::trinagulate_grid<CpuMC::PG, false>(grid,grid.getIsovalue(),vertData,normData);
        
        auto end = std::chrono::steady_clock::now();
        seq_res  = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for (const auto& v : vertData) c_seq += static_cast<size_t>(v.x + v.y + v.z);
        for (const auto& n : normData) c_seq += static_cast<size_t>(n.x + n.y + n.z);

    }
    
    {
        std::vector<glm::vec3> vertData, normData;
        auto start = std::chrono::steady_clock::now();
        CpuMC::trinagulate_grid<CpuMC::PG, true>(grid,grid.getIsovalue(),vertData,normData);
        
        auto end = std::chrono::steady_clock::now();
        omp_res = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        for (const auto& v : vertData) c_omp += static_cast<size_t>(v.x + v.y + v.z);
        for (const auto& n : normData) c_omp += static_cast<size_t>(n.x + n.y + n.z);
    }
    
    {
        auto [cp, ke, chsum] = CudaMC::test_time(grid,grid.getIsovalue());
        cuda_res_cop = cp;
        cuda_res_ker = ke;
        c_cuda = chsum;
    }

    std::cout << dim << ";" << seq_res << ";" << omp_res << ";" << cuda_res_ker << ";" << cuda_res_ker+cuda_res_cop << std::endl;
}