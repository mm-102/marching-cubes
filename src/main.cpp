#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>
#include <window_manager.hpp>
#include <camera.hpp>
#include <triangles.hpp>
#include <smooth_triangles.hpp>
#include <generator.hpp>
#include <marching_cubes.hpp>
#include <grid.hpp>

std::vector<glm::vec3> triangle_buf;
std::mutex triangle_buf_mut;
std::atomic_bool should_stop{false};
std::atomic_bool march_finished{false};

void march(Grid<float> &grid, float isovalue){
	MarchingCubes::triangulate_grid_to_vec(grid, isovalue, triangle_buf, triangle_buf_mut, should_stop);
	march_finished = true;
}

void use_triangle_buf(std::shared_ptr<Triangles> &triangles){
	std::lock_guard<std::mutex> lock(triangle_buf_mut);
	if(!triangle_buf.empty()){
		triangles->add_verticies(triangle_buf);
		triangle_buf.clear();
	}
}

void use_triangle_buf_smooth(std::shared_ptr<SmoothTriangles> &triangles){
	std::lock_guard<std::mutex> lock(triangle_buf_mut);
	if(!triangle_buf.empty()){
		triangles->add_verticies(triangle_buf);
		triangle_buf.clear();
	}
}

int main(){
	WindowManager windowManager(1024, 720, "Marching Cubes");

	if(!windowManager.init()){
		exit(EXIT_FAILURE);
	}

	Camera camera(glm::vec3(0.0f,0.0f,-50.0f), glm::vec3(0.0f,0.0f,1.0f), 0.873f, 200.0f, windowManager.get_size_ratio());

	std::cout << "start gen" << std::endl;
	Generator gen(glm::uvec3(100));

	glm::vec3 sphereCenter = glm::vec3(50.0f);
	// Grid<float> sphereGrid = gen.genSphere(sphereCenter, 25.0f);
	Grid<float> sphereGrid = gen.genTorus(sphereCenter, 8.0f, 25.0f);
	std::cout << "sphere generated" << std::endl;

	glm::mat4 M = glm::translate(glm::mat4(1.0f), -sphereCenter);
	// std::shared_ptr<Triangles> triangles(new Triangles(80000, M));
	// windowManager.add_object(triangles);

	std::shared_ptr<SmoothTriangles> triangles(new SmoothTriangles(80000, M));
	windowManager.add_object(triangles);

	// std::vector<glm::vec3> data = MarchingCubes::trinagulate_grid(sphereGrid, 0.0f);
	// triangles->add_verticies(data);
	// triangles->smooth();
	
	// std::vector<glm::vec3> test_data = MarchingCubes::trinagulate_grid(sphereGrid, 0.0f);
	// std::cout << "sphere triangulated " << test_data.size() << std::endl;
		
	// std::vector<glm::vec3> old_test = {
	// 	{0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
	// 	{0.0f, 0.0f, 2.0f}, {1.0f, 1.0f, 2.0f}, {0.0f, 1.0f, 2.0f},
	// 	{-1.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}
	// };

	// triangles -> add_verticies(test_data);

	windowManager.attach_resize_callback([&](int w, int h, int ratio){camera.updateWindowRatio(ratio);});
	windowManager.attach_mouse_pos_callback([&](double xrel, double yrel){camera.handle_mouse_pos_event(xrel,yrel);});
	windowManager.attach_mouse_button_callback([&](int button, int action, int mods){camera.handle_mouse_button_event(button,action,mods);});
	windowManager.attach_scroll_callback([&](double xoff, double yoff){camera.handle_scroll_event(xoff,yoff);});
	windowManager.attach_key_callback([&](int key,int scancode, int action, int mods){camera.handle_key_event(key,action,mods);});

	std::thread marching_thread(march, std::ref(sphereGrid), 0.0f);
	
	float delta = 0.0f;
	bool smoothed = false;
	while(!windowManager.should_close()){
		delta = windowManager.getDelta();
		camera.update(delta);
		windowManager.draw_scene(camera);
		windowManager.poll_events();

		if(!smoothed){
			use_triangle_buf_smooth(triangles);
			if(march_finished){
				triangles->smooth();
				smoothed = true;
			}
		}
	}
	should_stop = true;
	marching_thread.join();

    return EXIT_SUCCESS;
}