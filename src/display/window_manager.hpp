#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cstdio>
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include "simple_renderer.hpp"
#include "camera.hpp"

class WindowManager{
    glm::ivec2 window_size;
    float window_ratio;
    const std::string title;
    GLFWwindow* window;
    std::unique_ptr<SimpleRenderer> renderer;
    std::vector<std::shared_ptr<RenderableObject>> objects;
    double last_mouse_pos[2]{0};

    static void error_callback(int error, const char* description);

    void key_callback(GLFWwindow* window, int key,
        int scancode, int action, int mods);
    void mouse_pos_callback(GLFWwindow* window, double xpos, double ypos);
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    void scroll_callback(GLFWwindow* window, double xoff, double yoff);
    void window_resize_callback(GLFWwindow* window, int width, int height);
    
    std::function<void(int,int,int,int)> attached_key_callback = [](int,int,int,int){};
    std::function<void(int,int,float)> attached_resize_callback = [](int,int,float){};
    std::function<void(int,int,int)> attached_mouse_button_callback = [](int,int,int){};
    std::function<void(double,double)> attached_scroll_callback = [](double,double){};
    std::function<void(double,double)> attached_mouse_pos_callback = [](double,double){};

public:
    WindowManager(int width, int heigth, const std::string title);
    WindowManager(glm::ivec2 size, const std::string title);
    ~WindowManager();
    bool init();
    bool should_close();
    void poll_events();
    float getDelta();
    
    
    glm::ivec2 get_size();
    void set_size(glm::ivec2 size);
    float get_size_ratio();

    void attach_key_callback(std::function<void(int,int,int,int)> callback);
    void attach_resize_callback(std::function<void(int,int,float)> callback);
    void attach_mouse_button_callback(std::function<void(int,int,int)> callback);
    void attach_scroll_callback(std::function<void(double,double)> callback);
    void attach_mouse_pos_callback(std::function<void(double,double)> callback);

    void draw_scene(Camera &camera);
    void add_object(std::shared_ptr<RenderableObject> obj);
};