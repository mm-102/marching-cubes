#include "cxxopts.hpp"
#include <vector>
#include <grid.hpp>
#include <generator.hpp>
#include <save_obj.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <chrono>
#include <mutex>
#include <atomic>
#include <thread>
#include <triangles.hpp>
#include <window_manager.hpp>
#include <camera.hpp>
#include <CpuMC.hpp>
#include <CudaMC.hpp>

std::vector<glm::vec3> vertBuf, normBuf;
std::mutex mut;
std::atomic_bool should_stop{false};
std::atomic_bool march_finished{false};

void march(Grid<float> &grid, float isovalue, double delay, bool use_grad, const std::string mode){
    if(use_grad){
        if(mode == "omp")
            CpuMC::trinagulate_grid_mut<CpuMC::PG,true>(grid,isovalue,vertBuf,normBuf, mut, should_stop, delay);
        else if(mode == "seq")
            CpuMC::trinagulate_grid_mut<CpuMC::PG,false>(grid,isovalue,vertBuf,normBuf, mut, should_stop, delay);
        else if(mode == "cuda")
            std::cerr << "Cuda is not avaliable in animate mode" << std::endl;
        else{
            std::cerr << "Incorrect mode, use seq, omp or cuda" << std::endl;
        }
    }
    else{
        if(mode == "omp")
            CpuMC::trinagulate_grid_mut<CpuMC::P,true>(grid,isovalue,vertBuf,normBuf, mut, should_stop, delay);
        else if(mode == "seq")
            CpuMC::trinagulate_grid_mut<CpuMC::P,false>(grid,isovalue,vertBuf,normBuf, mut, should_stop, delay);
        else if(mode == "cuda")
            std::cerr << "Cuda is not avaliable in animate mode" << std::endl;
        else{
            std::cerr << "Incorrect mode, use seq, omp or cuda" << std::endl;
        }
    }
	march_finished = true;
}

void use_buf(std::shared_ptr<Triangles> &triangles){
	std::lock_guard<std::mutex> lock(mut);
	if(!vertBuf.empty()){
		triangles->add_verticies(vertBuf, normBuf);
		vertBuf.clear();
		normBuf.clear();
	}
}

int main(int argc, const char *argv[]){

    float isovalue;
    bool use_grad;
    bool animate;
    double delay;

    cxxopts::Options options("Marching Cubes", "With visualisation and animation");

    options.add_options()
        ("s,smooth", "Use gradient smoothing", cxxopts::value<bool>(use_grad)->default_value("false"))
        ("d,dimensions", "Dimensions of generated shape", cxxopts::value<std::vector<float>>())
        ("g,generate", "generate selected shape", cxxopts::value<std::string>())
        ("i,input", "load grid from file", cxxopts::value<std::string>())
        ("v,value,isovalue", "set custom isovalue", cxxopts::value<float>(isovalue)->default_value("0.0"))
        ("a,animate", "Show mesh during triangulation (slow)", cxxopts::value<bool>(animate)->default_value("false"))
        ("t,time,delay_time", "Delay between iterations when animating", cxxopts::value<double>(delay)->default_value("0.0"))
        ("m,mode", "use seq, omp or cuda", cxxopts::value<std::string>()->default_value("omp"))
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

    
    glm::mat4 M = glm::translate(glm::mat4(1.0f), -glm::vec3(grid.getSize()) * 0.5f);
    
    std::vector<glm::vec3> vertData, normData;
    std::cout << "Grid prepared, starting triangulation in mode: " << (use_grad ? "smooth" : "flat") << std::endl;
    std::cout << "Using isovalue: " << isovalue << std::endl;
    
    WindowManager windowManager(1024, 720, "Marching Cubes");
	if(!windowManager.init()){
        std::cerr << "Error initializing window, aborting" << std::endl;
		exit(EXIT_FAILURE);
	}
    
    const glm::vec3 size = glm::vec3(grid.getSize());
    const float viev_range = std::max(size.x, std::max(size.y,size.z)) * 2.0f;
    Camera camera(glm::vec3(0.0f), 1.5f * size.z, 0.873f, viev_range, windowManager.get_size_ratio());

    windowManager.attach_resize_callback([&](int w, int h, int ratio){camera.updateWindowRatio(ratio);});
	windowManager.attach_mouse_pos_callback([&](double xrel, double yrel){camera.handle_mouse_pos_event(xrel,yrel);});
	windowManager.attach_mouse_button_callback([&](int button, int action, int mods){camera.handle_mouse_button_event(button,action,mods);});
	windowManager.attach_scroll_callback([&](double xoff, double yoff){camera.handle_scroll_event(xoff,yoff);});
	windowManager.attach_key_callback([&](int key,int scancode, int action, int mods){camera.handle_key_event(key,action,mods);});

    std::shared_ptr<Triangles> triangles;

    auto start = std::chrono::high_resolution_clock::now();
    auto on_finished = [&](){
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Mesh triangulated in [ms] : " << duration.count() << std::endl;
    };

    const std::string mode = result["mode"].as<std::string>();
    std::cout << "Implementation: " << mode << std::endl;
    std::thread marching_thread;
    if(!animate){
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

        triangles = std::make_shared<Triangles>(vertData.size(),M);
        windowManager.add_object(triangles);
        triangles->add_verticies(vertData,normData);
        on_finished();
    }
    else{
        const size_t init_res = static_cast<size_t>(size.x * size.y * size.z / 9);
        triangles = std::make_shared<Triangles>(init_res,M);
        windowManager.add_object(triangles);
        marching_thread = std::thread(march, std::ref(grid), isovalue, delay, use_grad, mode);
    }

    float delta = 0.0f;
    bool finished = false;
    while(!windowManager.should_close()){
        delta = windowManager.getDelta();
        camera.update(delta);
        windowManager.draw_scene(camera);
        windowManager.poll_events();
        if(animate && !finished){
            finished = march_finished;
            use_buf(triangles);
            if(finished)
                on_finished();
        }
    }
    should_stop = true;
    if(animate)
        marching_thread.join();
    exit(EXIT_SUCCESS);
}