#include "window_manager.hpp"

#include <iostream>

WindowManager::WindowManager(glm::ivec2 size, const std::string title) : window_size(size), window_ratio((float)size.x / (float)size.y), title(title){};
WindowManager::WindowManager(int width, int heigth, const std::string title) : window_size(width, heigth), window_ratio((float)width / (float)heigth), title(title){};

WindowManager::~WindowManager(){
    glfwDestroyWindow(window);
    glfwTerminate();
}

void WindowManager::error_callback(int error, const char* description){
    fputs(description, stderr);
}

bool WindowManager::init(){
    glfwSetErrorCallback(WindowManager::error_callback);

    if(!glfwInit()){
        std::cerr << "Can't initialize GLFW" << std::endl;
        return false;
    }

    window = glfwCreateWindow(window_size.x, window_size.y, title.c_str(), NULL, NULL);

    glfwMakeContextCurrent(window);
	glfwSwapInterval(1); //During vsync wait for the first refresh

    GLenum err;
	if ((err = glewInit()) != GLEW_OK) {
        std::cerr << "Can't initialize GLEW: " << glewGetErrorString(err) << std::endl;
		return false;
	}

    renderer = std::unique_ptr<SimpleRenderer>(new SimpleRenderer());

    glfwSetWindowUserPointer(window, this);

    glfwSetKeyCallback(window, [](GLFWwindow* w, int k, int sc, int a, int m){
        static_cast<WindowManager*>(glfwGetWindowUserPointer(w))->key_callback(w,k,sc,a,m);
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* w, double x, double y){
        static_cast<WindowManager*>(glfwGetWindowUserPointer(w))->mouse_pos_callback(w,x,y);
    });

    glfwSetMouseButtonCallback(window, [](GLFWwindow* w, int b, int a, int m){
        static_cast<WindowManager*>(glfwGetWindowUserPointer(w))->mouse_button_callback(w,b,a,m);
    });

    glfwSetWindowSizeCallback(window, [](GLFWwindow* w, int wi, int he){
        static_cast<WindowManager*>(glfwGetWindowUserPointer(w))->window_resize_callback(w,wi,he);
    });

    glfwSetTime(0);

    glClearColor(0.2, 0.2, 0.2, 1.0);
    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return true;
}

bool WindowManager::should_close(){
    return glfwWindowShouldClose(window);
}
void WindowManager::poll_events(){
    glfwPollEvents();
}

glm::ivec2 WindowManager::get_size(){
    return window_size;
}

void WindowManager::set_size(glm::ivec2 size){
    window_size = size;
    window_ratio = static_cast<float>(size.x) / static_cast<float>(size.y);
}

float WindowManager::get_size_ratio(){
    return window_ratio;
}

void WindowManager::key_callback(GLFWwindow* window, int key,
    int scancode, int action, int mods){
    attached_key_callback(key, scancode, action, mods);
}

void WindowManager::mouse_pos_callback(GLFWwindow* window, double xpos, double ypos){
    glm::vec2 half_size = glm::vec2(window_size) * 0.5f;
    xpos -= half_size.x;
    ypos -= half_size.y;
    attached_mouse_pos_callback(xpos-last_mouse_pos[0], ypos-last_mouse_pos[1]);
    last_mouse_pos[0] = xpos;
    last_mouse_pos[1] = ypos;
}

void WindowManager::mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    attached_mouse_button_callback(button, action, mods);
}

void WindowManager::window_resize_callback(GLFWwindow* window, int width, int height){
    if (height == 0) return;
    set_size(glm::ivec2(width, height));
    attached_resize_callback(width,height,window_ratio);
}

void WindowManager::attach_key_callback(std::function<void(int,int,int,int)> callback){
    attached_key_callback = callback;
}
void WindowManager::attach_resize_callback(std::function<void(int,int,float)> callback){
    attached_resize_callback = callback;
}
void WindowManager::attach_mouse_button_callback(std::function<void(int,int,int)> callback){
    attached_mouse_button_callback = callback;
}
void WindowManager::attach_mouse_pos_callback(std::function<void(double,double)> callback){
    attached_mouse_pos_callback = callback;
}

void WindowManager::draw_scene(Camera &camera){
    renderer->useCamera(camera);
    for(auto obj : objects){
        renderer->draw(obj);
    }
    glfwSwapBuffers(window);
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);
    glClearColor(0.2, 0.2, 0.2, 1.0);
}

void WindowManager::add_object(std::shared_ptr<RenderableObject> obj){
    objects.push_back(obj);
}

// return glfw time since last call of this function
float WindowManager::getDelta(){
    float delta = glfwGetTime();
    glfwSetTime(0);
    return delta;
}