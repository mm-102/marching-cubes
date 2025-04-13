#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "shaderprogram.hpp"

class Camera{
    glm::vec4 pos;
    glm::vec4 dir;
    glm::quat rot;
    glm::vec2 rotxy;
    glm::vec3 lookCenter;

    glm::vec2 ang_vel_dir;
    float ang_vel_v;

    float dist;
    float scroll_mul;

    float mouse_sensivity;
    bool rot_drag;

    bool move_drag;
    glm::vec3 move_drag_start;

    float fov;
    float far;

    glm::mat4 P;
    glm::mat4 V;

public:
    Camera(glm::vec3 pos, glm::vec3 dir, float fov, float far, float ratio);
    void useCamera(ShaderProgram &shaderProgram);
    void updateWindowRatio(float ratio);
    bool handle_key_event(int key, int action, int mods);
	void handle_mouse_pos_event(double xrel, double yrel);
	void handle_mouse_button_event(int button, int action, int mods);
	void handle_scroll_event(double xoff, double yoff);
	void update(float delta);

    glm::vec3 get_pos();
    glm::vec3 get_dir();
    glm::quat get_rot();
};
