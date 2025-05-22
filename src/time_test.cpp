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

int main(int argc, const char *argv[]){

    bool use_grad;
    unsigned min_dim;
    unsigned max_dim;
    unsigned num_tests;
    int64_t rep;
    std::string out_file;

    cxxopts::Options options("Marching cube testing", "measure time");

    options.add_options()
        ("min","min_dim", cxxopts::value<unsigned>(min_dim)->default_value("10"))
        ("max","max_dim", cxxopts::value<unsigned>(max_dim)->default_value("500"))
        ("n","num_tests", cxxopts::value<unsigned>(num_tests)->default_value("20"))
        ("r","rep_per_test", cxxopts::value<int64_t>(rep)->default_value("5"))
        ("s","smooth", cxxopts::value<bool>(use_grad)->default_value("false"))
        ("o","out", cxxopts::value<std::string>(out_file)->default_value("out.csv"))
    ;

    auto result = options.parse(argc, argv);

    {
        std::ofstream file(out_file);
        file << "N;dim^3;seq;omp;cuda;" << std::endl;
        file.close();
        std::cout << "Writing res to: " << out_file << std::endl;
    }

    CudaMC::setConstMem();
    for(unsigned test_no = 0; test_no < num_tests; test_no++){
        const float p = static_cast<float>(test_no) / static_cast<float>(num_tests);
        const float dif = static_cast<float>(max_dim - min_dim) * p;
        const int dim = min_dim + static_cast<unsigned>(dif);

        const float s = static_cast<float>(dim) / 10.0f;

        
        int64_t seq_res = 0, omp_res = 0, cuda_res = 0;
        volatile size_t c_seq = 0, c_omp = 0, c_cuda = 0;
        
        std::cout << "N: " << test_no << " / " << num_tests << "\t";
        std::cout << "dim: " << dim << "\t";
        
        for(int r = 0; r < rep; r++){

            Grid<float> grid = gen::perlin(glm::uvec3(dim), s);
            {
                std::vector<glm::vec3> vertData, normData;
                auto start = std::chrono::high_resolution_clock::now();
                if(use_grad)
                    CpuMC::trinagulate_grid<CpuMC::PG, false>(grid,grid.getIsovalue(),vertData,normData);
                else
                    CpuMC::trinagulate_grid<CpuMC::P, false>(grid,grid.getIsovalue(),vertData,normData);
                
                auto end = std::chrono::high_resolution_clock::now();
                seq_res += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                size_t checksum = 0;
                for (const auto& v : vertData) checksum += static_cast<size_t>(v.x + v.y + v.z);
                for (const auto& n : normData) checksum += static_cast<size_t>(n.x + n.y + n.z);
                c_seq += checksum;

            }
            
            {
                std::vector<glm::vec3> vertData, normData;
                auto start = std::chrono::high_resolution_clock::now();
                if(use_grad)
                    CpuMC::trinagulate_grid<CpuMC::PG, true>(grid,grid.getIsovalue(),vertData,normData);
                else
                    CpuMC::trinagulate_grid<CpuMC::P, true>(grid,grid.getIsovalue(),vertData,normData);
                
                auto end = std::chrono::high_resolution_clock::now();
                omp_res += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                size_t checksum = 0;
                for (const auto& v : vertData) checksum += static_cast<size_t>(v.x + v.y + v.z);
                for (const auto& n : normData) checksum += static_cast<size_t>(n.x + n.y + n.z);
                c_omp += checksum;

            }
            
            {
                std::vector<glm::vec3> vertData, normData;
                auto start = std::chrono::high_resolution_clock::now();
                if(use_grad)
                    CudaMC::trinagulate_grid<CudaMC::PG>(grid,grid.getIsovalue(),vertData,normData);
                else
                    CudaMC::trinagulate_grid<CudaMC::P>(grid,grid.getIsovalue(),vertData,normData);
                
                auto end = std::chrono::high_resolution_clock::now();
                cuda_res += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

                size_t checksum = 0;
                for (const auto& v : vertData) checksum += static_cast<size_t>(v.x + v.y + v.z);
                for (const auto& n : normData) checksum += static_cast<size_t>(n.x + n.y + n.z);
                c_cuda += checksum;

            }
        }

        std::cout << "sum: " << c_seq << " " << c_omp << " " << c_cuda << std::endl;
        
        seq_res /= rep;
        omp_res /= rep;
        cuda_res /= rep;

        std::ofstream file(out_file, std::ios::app);

        file << test_no << ";" << dim << ";" << seq_res << ";" << omp_res << ";" << cuda_res << ";" << std::endl;

        file.close();
    }
}