#include "camera.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 lookCenter, float dist, float fov, float far, float ratio) : 
    pos(lookCenter.x,lookCenter.y,lookCenter.z - dist, 1.0f), 
    dir(0.0f,0.0f,1.0f, 0.0f), 
    rotxy(0.0f),
    lookCenter(lookCenter),
    ang_vel_dir(0.0f), 
    dist(dist),
    scroll_mul(1.0f),
    rot_drag(false),
    move_drag(false),
    move_drag_start(0.0f),
    fov(fov), far(far)
{
    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    mouse_sensivity = 0.005;
    ang_vel_v = 5.0f;
    V = glm::lookAt(glm::vec3(pos), lookCenter, glm::vec3(0.0f,1.0f,0.0f));
    updateWindowRatio(ratio);
}

void Camera::updateWindowRatio(float ratio){
    P = glm::perspective(fov, ratio, 1.0f, far);
}

void Camera::useCamera(ShaderProgram &shaderProgram){
    shaderProgram.use();
    glUniformMatrix4fv(shaderProgram.u("P"), 1, false, glm::value_ptr(P));
	glUniformMatrix4fv(shaderProgram.u("V"), 1, false, glm::value_ptr(V));
	glUniform3fv(shaderProgram.u("uCamera"),1,glm::value_ptr(glm::vec3(pos)));
}

bool Camera::handle_key_event(int key, int action, int mods){
    bool event_consumed = false;
    
    if(action == GLFW_PRESS){
        event_consumed = true;
        switch(key){
            case GLFW_KEY_W:
                ang_vel_dir.x = 1.0f;
                break;
            case GLFW_KEY_S:
                ang_vel_dir.x = -1.0f;
                break;
            case GLFW_KEY_A:
                ang_vel_dir.y = -1.0f;
                break;
            case GLFW_KEY_D:
                ang_vel_dir.y = 1.0f;
                break;
            case GLFW_KEY_R:
                rotxy = glm::vec2(0.0f);
                lookCenter = glm::vec3(0.0f);
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
                ang_vel_dir.x = 0.0f;
				break;
            case GLFW_KEY_A:
            case GLFW_KEY_D:
                ang_vel_dir.y = 0.0f;
                break;
            default:
                event_consumed = false;
        }
        return event_consumed;
    }

    return event_consumed;
}

void Camera::handle_mouse_button_event(int button, int action, int mods){
    if(action == GLFW_PRESS){
        switch(button){
            case GLFW_MOUSE_BUTTON_1:
            case GLFW_MOUSE_BUTTON_2:
                rot_drag = true;
                break;
            case GLFW_MOUSE_BUTTON_3:
                move_drag = true;
                move_drag_start = lookCenter;
        }
    }

    else if(action == GLFW_RELEASE){
        switch(button){
            case GLFW_MOUSE_BUTTON_1:
            case GLFW_MOUSE_BUTTON_2:
                rot_drag = false;
                break;
            case GLFW_MOUSE_BUTTON_3:
                move_drag = false;
        }
    }
}

void Camera::handle_scroll_event(double xoff, double yoff){
    dist = glm::clamp(dist - static_cast<float>(yoff) * scroll_mul * std::log(dist), 10.0f, 1000.0f);
}

void Camera::handle_mouse_pos_event(double xrel, double yrel){
    if(rot_drag){
        rotxy += glm::vec2(yrel, xrel) * mouse_sensivity;
    }
    else if(move_drag){ // kind of messy
        glm::vec3 left = glm::vec3(-dir.z, dir.y, dir.x);
        lookCenter += left * static_cast<float>(xrel) * mouse_sensivity;
        lookCenter -= glm::cross(glm::vec3(dir), left) * static_cast<float>(yrel) * mouse_sensivity;
    }
}

void Camera::update(float delta){
    rotxy += ang_vel_dir * ang_vel_v * delta;

    constexpr static float pi = glm::pi<float>();
    rotxy.x = glm::clamp(rotxy.x, -pi*0.499f, pi*0.499f);

    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    rot = glm::rotate(rot, rotxy.y, glm::vec3(0.0f,1.0f,0.0f));
    rot = glm::rotate(rot, rotxy.x, glm::vec3(1.0f,0.0f,0.0f));

    dir = glm::mat4_cast(glm::normalize(rot)) * glm::vec4(0.0f,0.0f,1.0f,0.0f);

    V = glm::lookAt(lookCenter - glm::vec3(-dir) * dist, lookCenter, glm::vec3(0.0f, 1.0f, 0.0f));
    pos = glm::vec4(lookCenter, 0.0f) - dir;
}

glm::vec3 Camera::get_pos(){
    return glm::vec3(pos);
}
glm::vec3 Camera::get_dir(){
    return glm::vec3(dir);
}
glm::quat Camera::get_rot(){
    return rot;
}