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

}

void WindowManager::mouse_pos_callback(GLFWwindow* window, double xpos, double ypos){

}

void WindowManager::mouse_button_callback(GLFWwindow* window, int button, int action, int mods){

}

void WindowManager::window_resize_callback(GLFWwindow* window, int width, int height){
    if (height == 0) return;
    set_size(glm::ivec2(width, height));
}