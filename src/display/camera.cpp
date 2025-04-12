#include "camera.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 pos, glm::vec3 dir, float fov, float far, float ratio) : pos(pos, 1.0f), dir(dir, 0.0f), fov(fov), far(far), vel_dir(0.0f){
    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    rotx = 0;
    roty = 0;
    mouse_sensivity = 0.001;
    max_vel = 10.0f;
    V = glm::lookAt(pos, pos+dir, glm::vec3(0.0f,1.0f,0.0f));
    updateWindowRatio(ratio);
}

void Camera::updateWindowRatio(float ratio){
    P = glm::perspective(fov, ratio, 1.0f, far);
}

void Camera::set_mode(Mode mode){
    this->mode = mode;
}

void Camera::useCamera(ShaderProgram &shaderProgram){
    glUniformMatrix4fv(shaderProgram.u("P"), 1, false, glm::value_ptr(P));
	glUniformMatrix4fv(shaderProgram.u("V"), 1, false, glm::value_ptr(V));
	glUniform3fv(shaderProgram.u("uCamera"),1,glm::value_ptr(glm::vec3(pos)));
}

bool Camera::handle_key_event(int key, int action, int mods){
    bool event_consumed = false;
    if (mode == Camera::Mode::CENTER)
        return event_consumed;
    
    if(action == GLFW_PRESS){
        event_consumed = true;
        switch(key){
            case GLFW_KEY_W:
                vel_dir.z = 1.0f;
                break;
            case GLFW_KEY_S:
                vel_dir.z = -1.0f;
                break;
            case GLFW_KEY_A:
                vel_dir.x = -1.0f;
                break;
            case GLFW_KEY_D:
                vel_dir.x = 1.0f;
                break;
            case GLFW_KEY_SPACE:
                vel_dir.y = 1.0f;
                break;
            case GLFW_KEY_LEFT_SHIFT:
                vel_dir.y = -1.0f;
                break;
            default:
                event_consumed = false;
        }
        return event_consumed;
    }

    if(action == GLFW_RELEASE){
        event_consumed = true;
        switch(key){
            case GLFW_KEY_W:
			case GLFW_KEY_S:
				vel_dir.z = 0.0f;
				break;
            case GLFW_KEY_A:
            case GLFW_KEY_D:
                vel_dir.x = 0.0f;
                break;
            case GLFW_KEY_SPACE:
            case GLFW_KEY_LEFT_SHIFT:
                vel_dir.y = 0.0f;
                break;
            default:
                event_consumed = false;
        }
        return event_consumed;
    }

    return event_consumed;
}

void Camera::handle_mouse_pos_event(double xrel, double yrel){
    roty += xrel * mouse_sensivity;
    
    const static double pi = glm::pi<double>();
	rotx = glm::clamp(rotx - yrel * mouse_sensivity,-pi*0.499,pi*0.499);

    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    rot = glm::rotate(rot, roty, glm::vec3(0.0f,1.0f,0.0f));
    rot = glm::rotate(rot, rotx, glm::vec3(1.0f,0.0f,0.0f));

    dir = glm::mat4_cast(glm::normalize(rot)) * glm::vec4(0.0f,0.0f,1.0f,0.0f);

    switch(mode){
        case Camera::Mode::FREE:
            V = glm::lookAt(glm::vec3(pos), glm::vec3(pos + dir), glm::vec3(0.0f, 1.0f, 0.0f));
            break;

        case Camera::Mode::CENTER:
        default:
            V = glm::lookAt(glm::vec3(-dir), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    }
}

void Camera::update(float delta){
    if(vel_dir == glm::vec3(0))
        return;

    glm::vec3 off(0.0f);

    off += glm::normalize(glm::vec3(dir.x,0.0f,dir.z)) * vel_dir.z;
    off += glm::normalize(glm::vec3(-dir.z, 0.0f, dir.x)) * vel_dir.x;
    if(off != glm::vec3(0.0f)) off = glm::normalize(off);
    off += glm::vec3(0.0f, 1.0f, 0.0f) * vel_dir.y;
    pos += glm::vec4(off, 0.0f) * delta * max_vel;

    
    V = glm::lookAt(glm::vec3(pos), glm::vec3(pos+dir), glm::vec3(0.0f,1.0f,0.0f));
}

glm::vec3 Camera::get_pos(){
    return glm::vec3(pos);
}
glm::vec3 Camera::get_dir(){
    return glm::vec3(pos);
}
glm::quat Camera::get_rot(){
    return rot;
}