#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <thread>
#include <functional>
#include <window_manager.hpp>
#include <camera.hpp>
#include <triangles.hpp>
#include <generator.hpp>
#include <marching_cubes.hpp>
#include <grid.hpp>

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
	Grid<float> sphereGrid = gen.genTorus(sphereCenter, 5.0f, 25.0f);
	std::cout << "sphere generated" << std::endl;

	glm::mat4 M = glm::translate(glm::mat4(1.0f), -sphereCenter);
	std::shared_ptr<Triangles> triangles(new Triangles(80000, M));
	windowManager.add_object(triangles);

	// std::thread marching_thread(march, std::ref(triangles), std::ref(sphereGrid), 0.0f);
	// std::thread marching_thread(march2, std::ref(windowManager));
	
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

	std::cout << "Started triangulating" << std::endl;
	float delta = 0.0f;
	int vertC = 0;
	MarchingCubes::triangulate_grid_steps(sphereGrid, 0.0f, [&](std::vector<glm::vec3> &data){
		delta = windowManager.getDelta();
		triangles-> add_verticies(data);
		camera.update(delta);
		windowManager.draw_scene(camera);
		windowManager.poll_events();
		vertC += data.size();
		return windowManager.should_close();
	});
	std::cout << "Finished triangulating: " << vertC / 3 << " trinagles" << std::endl;

	while(!windowManager.should_close()){
		delta = windowManager.getDelta();
		camera.update(delta);
		windowManager.draw_scene(camera);
		windowManager.poll_events();
	}

    return EXIT_SUCCESS;
}