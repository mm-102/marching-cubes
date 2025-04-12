#version 400

layout (location=0) in vec3 aPos;

out vec3 vPos;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main(){
    vPos = vec3(M * vec4(aPos,1.0));

    gl_Position = P * V * M * vec4(aPos, 1.0);
}
