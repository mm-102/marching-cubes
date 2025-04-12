#include <cstdlib>
#include <cstdio>
#include <memory>
#include <vector>
#include <functional>
#include <window_manager.hpp>
#include <camera.hpp>
#include <triangles.hpp>

int main(){
	WindowManager windowManager(500, 500, "Marching Cubes");

	if(!windowManager.init()){
		exit(EXIT_FAILURE);
	}

	Camera camera(glm::vec3(0.0f,0.0f,-10.0f), glm::vec3(0.0f,0.0f,1.0f), 0.873f, 50.0f, windowManager.get_size_ratio(), Camera::Mode::FREE);

	std::shared_ptr<RenderableObject> triangles(new Triangles(3 * 10));
	windowManager.add_object(triangles);

	std::vector<float> test_data = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	static_cast<Triangles*>(triangles.get())->add_verticies(test_data);

	windowManager.attach_resize_callback([&](int w, int h, int ratio){camera.updateWindowRatio(ratio);});
	windowManager.attach_mouse_pos_callback([&](double xrel, double yrel){camera.handle_mouse_pos_event(xrel,yrel);});

	while(!windowManager.should_close()){
		windowManager.draw_scene(camera);
		windowManager.poll_events();
	}

    return EXIT_SUCCESS;
}