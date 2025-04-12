#pragma once

#include "shaderprogram.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class RenderableObject{
public:
    virtual glm::mat4 getModelMatrix() { return glm::mat4(1.0f); };
    virtual glm::vec4 getModulate() { return glm::vec4(1.0f); };
    virtual glm::vec4 getSelfModulate() { return glm::vec4(1.0f); };
    virtual void draw(ShaderProgram &shader) {};
    virtual std::vector<std::shared_ptr<RenderableObject>> getChildren(){ return {};};
};