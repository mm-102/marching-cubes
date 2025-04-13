#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "shaderprogram.hpp"

class Camera{
public:
    enum class Mode{
        CENTER,
        FREE
    };
private:
    glm::vec4 pos;
    glm::vec4 dir;
    glm::quat rot;
    float rotx, roty;

    glm::vec3 lin_vel_dir;
    float lin_vel_v;
    glm::vec2 ang_vel_dir;
    float ang_vel_v;

    float mouse_sensivity;

    float fov;
    float far;

    glm::mat4 P;
    glm::mat4 V;

    Mode mode;

public:
    Camera(glm::vec3 pos, glm::vec3 dir, float fov, float far, float ratio, Mode mode = Mode::CENTER);
    void useCamera(ShaderProgram &shaderProgram);
    void updateWindowRatio(float ratio);
    bool handle_key_event(int key, int action, int mods);
	void handle_mouse_pos_event(double xrel, double yrel);
	void update(float delta);
    void set_mode(Mode mode);

    glm::vec3 get_pos();
    glm::vec3 get_dir();
    glm::quat get_rot();
};
