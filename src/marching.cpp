#include "cxxopts.hpp"
#include <vector>
#include <CudaMC.hpp>
#include <grid.hpp>
#include <generator.hpp>
#include <save_obj.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <chrono>
#include <CpuMC.hpp>

int main(int argc, const char *argv[]){

    float isovalue;
    bool use_grad;
    std::string out_file;
    bool no_save;

    cxxopts::Options options("Marching Cubes", "With no visualisation");

    options.add_options()
        ("s,smooth", "Use gradient smoothing", cxxopts::value<bool>(use_grad)->default_value("false"))
        ("d,dimensions", "Dimensions of generated shape", cxxopts::value<std::vector<float>>())
        ("g,generate", "generate selected shape", cxxopts::value<std::string>())
        ("o,output", "output .obj file", cxxopts::value<std::string>(out_file)->default_value("out.obj"))
        ("n,no_save", "don't save results", cxxopts::value<bool>(no_save)->default_value("false"))
        ("i,input", "load grid from file", cxxopts::value<std::string>())
        ("v,value,isovalue", "set custom isovalue", cxxopts::value<float>(isovalue)->default_value("0.0"))
        ("m,mode", "use seq, omp or cuda", cxxopts::value<std::string>()->default_value("cuda"))
    ;

    auto result = options.parse(argc, argv);

    Grid<float> grid;

    if(result.count("generate")){
        const std::string shape = result["generate"].as<std::string>();
        auto &dims = result["dimensions"].as<std::vector<float>>();

        if(shape == "torus"){
            float r_minor, r_major;
            if(dims.size() == 1){
                r_minor = dims.at(0);
                r_major = 3.0f * r_minor;
            }
            else if(dims.size() == 2){
                r_minor = dims.at(0);
                r_major = dims.at(1);
            }
            else {
                std::cerr << "Generator torus incorrect dimentions!" << std::endl;
                exit(EXIT_FAILURE);
            }
            std::cout << "Generator shape: torus " << r_minor << " " << r_major << std::endl;
            grid = gen::torus(r_minor, r_major);
        }
        else if(shape == "sphere"){
            float r;
            if(dims.size() == 1){
                r = dims.at(0);
            }
            else {
                std::cerr << "Generator sphere incorrect dimentions!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cout << "Generator shape: sphere " << r << std::endl;
            grid = gen::sphere(r);
        }
        else if(shape == "perlin"){
            float sc;
            glm::uvec3 s;
            if(dims.size() == 2){
                sc = dims.at(0);
                s = glm::uvec3(dims.at(1));
            }
            else if(dims.size() == 4){
                sc = dims.at(0);
                s = glm::uvec3(dims.at(1), dims.at(2), dims.at(3));
            }
            else {
                std::cerr << "Generator perlin incorrect dimentions!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::cout << "Generator shape: perlin " << s.x << " " << s.y << " " << s.z << std::endl;
            grid = gen::perlin(s, sc);
        }
        else {
            std::cerr << "Generator: incorrect generate shape!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if(result.count("input")){
        const std::string input = result["input"].as<std::string>();
        grid = gen::fromFile(input);
    }
    else {
        std::cerr << "no data to process specified!" << std::endl;
        exit(EXIT_FAILURE);
    }

    const std::string mode = result["mode"].as<std::string>();

    std::vector<glm::vec3> vertData, normData;
    std::cout << "Using isovalue: " << isovalue << std::endl;
    std::cout << "Grid prepared, starting triangulation in mode: " << (use_grad ? "smooth" : "flat") << std::endl;
    std::cout << "Implementation: " << mode << std::endl;

    CudaMC::setConstMem();
    auto start = std::chrono::high_resolution_clock::now();

    if(use_grad){
        if(mode == "omp")
            CpuMC::trinagulate_grid<CpuMC::PG,true>(grid,isovalue,vertData,normData);
        else if(mode == "seq")
            CpuMC::trinagulate_grid<CpuMC::PG,false>(grid,isovalue,vertData,normData);
        else if(mode == "cuda")
            CudaMC::trinagulate_grid<CudaMC::PG>(grid,isovalue,vertData,normData);
        else{
            std::cerr << "Incorrect mode, use seq, omp or cuda" << std::endl;
        }
    }
    else{
        if(mode == "omp")
            CpuMC::trinagulate_grid<CpuMC::P,true>(grid,isovalue,vertData,normData);
        else if(mode == "seq")
            CpuMC::trinagulate_grid<CpuMC::P,false>(grid,isovalue,vertData,normData);
        else if(mode == "cuda")
            CudaMC::trinagulate_grid<CudaMC::P>(grid,isovalue,vertData,normData);
        else{
            std::cerr << "Incorrect mode, use seq,omp or cuda" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Mesh triangulated in [ms] : " << duration.count() << std::endl;
    std::cout << vertData.size() << std::endl;

    if(!no_save)
        saveOBJ(out_file, vertData, normData);
    else
        std::cout << "Skipping saving to file" << std::endl;
    
    exit(EXIT_SUCCESS);
}