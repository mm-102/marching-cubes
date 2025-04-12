#pragma once

#include "shaderprogram.hpp"
#include "renderable.hpp"
#include <memory>

class SimpleRenderer{
    ShaderProgram spMain;

public:
    struct DrawMod {
    public:
        glm::mat4 rel;
        glm::mat4 model_mod;
        glm::vec4 modulate;
    };
    constexpr static DrawMod DefaultDraw(){
        return DrawMod{.rel = glm::mat4(1.0f), .model_mod=glm::mat4(1.0f), .modulate=glm::vec4(1.0f)};
    };

    SimpleRenderer();
    void draw(std::shared_ptr<RenderableObject> &obj, DrawMod mod = SimpleRenderer::DefaultDraw());

};