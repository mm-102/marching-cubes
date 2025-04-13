#include "camera.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 pos, glm::vec3 dir, float fov, float far, float ratio) : 
    pos(pos, 1.0f), 
    dir(dir, 0.0f), 
    rotxy(0.0f), 
    ang_vel_dir(0.0f), 
    dist(10.0f),
    mouse_drag(false),
    fov(fov), far(far)
{
    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    mouse_sensivity = 0.001;
    ang_vel_v = 5.0f;
    V = glm::lookAt(pos, pos+dir, glm::vec3(0.0f,1.0f,0.0f));
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
    if(action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_1){
        mouse_drag = true;
    }
    else if(action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_1){
        mouse_drag = false;
    }
}

void Camera::handle_mouse_pos_event(double xrel, double yrel){
    // roty += xrel * mouse_sensivity;
    
    // const static double pi = glm::pi<double>();
	// rotx = glm::clamp(rotx - yrel * mouse_sensivity,-pi*0.499,pi*0.499);

    // rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    // rot = glm::rotate(rot, roty, glm::vec3(0.0f,1.0f,0.0f));
    // rot = glm::rotate(rot, rotx, glm::vec3(1.0f,0.0f,0.0f));

    // dir = glm::mat4_cast(glm::normalize(rot)) * glm::vec4(0.0f,0.0f,1.0f,0.0f);

    // V = glm::lookAt(glm::vec3(-dir) * 10.0f, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    // pos = -dir;
}

void Camera::update(float delta){
    rotxy += ang_vel_dir * ang_vel_v * delta;

    rot = glm::quat(0.0f, 0.0f, 0.0f, 1.0f);
    rot = glm::rotate(rot, rotxy.y, glm::vec3(0.0f,1.0f,0.0f));
    rot = glm::rotate(rot, rotxy.x, glm::vec3(1.0f,0.0f,0.0f));

    dir = glm::mat4_cast(glm::normalize(rot)) * glm::vec4(0.0f,0.0f,1.0f,0.0f);

    V = glm::lookAt(glm::vec3(-dir) * dist, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    pos = -dir;
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