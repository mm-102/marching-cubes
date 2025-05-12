#include "cxxopts.hpp"
#include <vector>
#include <marching_cubes_flat.hpp>
#include <marching_cubes_grad.hpp>
#include <CudaMC.hpp>
#include <grid.hpp>
#include <generator.hpp>
#include <save_obj.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <chrono>

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
    ;

    auto result = options.parse(argc, argv);

    Grid<float> grid(0,0,0);

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
            }
            unsigned xz = static_cast<unsigned>(std::ceil(2.0f * (r_minor + r_major)) + 4);
            unsigned y = static_cast<unsigned>(std::ceil(2.0f * r_minor) + 4);
            Generator gen(xz,y,xz);
            std::cout << "Generator shape: torus " << r_minor << " " << r_major << std::endl;
            grid = gen.genTorus(glm::vec3(xz,y,xz) * 0.5f, r_minor, r_major);
        }
        else if(shape == "sphere"){
            float r;
            if(dims.size() == 1){
                r = dims.at(0);
            }
            else {
                std::cerr << "Generator sphere incorrect dimentions!" << std::endl;
            }
            unsigned xyz = static_cast<unsigned>(std::ceil(2.0f * r) + 4);
            Generator gen(xyz,xyz,xyz);
            std::cout << "Generator shape: sphere " << r << std::endl;
            grid = gen.genSphere(glm::vec3(xyz,xyz,xyz) * 0.5f, r);
        }
        else {
            std::cerr << "Generator: incorrect generate shape!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else if(result.count("input")){
        const std::string input = result["input"].as<std::string>();
        grid = Generator::fromFile(input);
    }
    else {
        std::cerr << "no data to process specified!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<glm::vec3> vertData, normData;
    std::cout << "Grid prepared, starting triangulation in mode: " << (use_grad ? "smooth" : "flat") << std::endl;
    std::cout << "Using isovalue: " << isovalue << std::endl;
    CudaMC::setConstMem();
    auto start = std::chrono::high_resolution_clock::now();

    if(use_grad)
        MarchingCubesGrad::trinagulate_grid(grid, isovalue, vertData, normData);
    else
        CudaMC::trinagulate_grid<CudaMC::Grad>(grid, isovalue, vertData, normData);
        // CudaMarchingCubes::trinagulate_grid_flat(grid, isovalue, vertData, normData);
        // MarchingCubesFlat::trinagulate_grid(grid, isovalue, vertData, normData);
    

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