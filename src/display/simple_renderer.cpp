#include "simple_renderer.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <battery/embed.hpp>

SimpleRenderer::SimpleRenderer() : spMain(b::embed<"res/shaders/main.vert">(), b::embed<"res/shaders/main.frag">()){}

void SimpleRenderer::draw(std::shared_ptr<RenderableObject> &obj, DrawMod mod){
    glm::mat4 M = obj->getModelMatrix();
    glm::vec4 obj_modulate = obj->getModulate();
    glm::vec4 obj_self_modulate = obj->getSelfModulate();

    spMain.use();
    glUniformMatrix4fv(spMain.u("M"), 1, false, glm::value_ptr(mod.rel * M * mod.model_mod));
	glUniform4fv(spMain.u("uModulate"), 1, glm::value_ptr(obj_modulate * obj_self_modulate));

    obj->draw(spMain);

    DrawMod children_draw_mod{
        .rel = mod.rel * M,
        .model_mod = mod.model_mod,
        .modulate = mod.modulate * obj_modulate
    };

    for(auto child : obj->getChildren()){
        draw(child, children_draw_mod);
    }
}

void SimpleRenderer::useCamera(Camera &camera){
    spMain.use();
    camera.useCamera(spMain);
}