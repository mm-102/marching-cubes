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

	Camera camera(glm::vec3(0.0f,0.0f,-10.0f), glm::vec3(0.0f,0.0f,1.0f), 0.873f, 50.0f, windowManager.get_size_ratio());

	std::shared_ptr<RenderableObject> triangles(new Triangles(3 * 10));
	windowManager.add_object(triangles);

	std::vector<float> test_data = {
		0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 2.0f, 1.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f,
		-1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
	static_cast<Triangles*>(triangles.get())->add_verticies(test_data);

	windowManager.attach_resize_callback([&](int w, int h, int ratio){camera.updateWindowRatio(ratio);});
	windowManager.attach_mouse_pos_callback([&](double xrel, double yrel){camera.handle_mouse_pos_event(xrel,yrel);});
	windowManager.attach_mouse_button_callback([&](int button, int action, int mods){camera.handle_mouse_button_event(button,action,mods);});
	windowManager.attach_key_callback([&](int key,int scancode, int action, int mods){camera.handle_key_event(key,action,mods);});

	float delta = 0.0f;
	while(!windowManager.should_close()){
		delta = windowManager.getDelta();
		camera.update(delta);
		windowManager.draw_scene(camera);
		windowManager.poll_events();
	}

    return EXIT_SUCCESS;
}