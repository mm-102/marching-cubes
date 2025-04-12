#version 400

out vec4 fragColor;

in vec3 vPos;

uniform vec3 uCamera;
uniform vec4 uModulate = vec4(1.0,0.0,0.0,1.0);

void main(){
    fragColor = uModulate;
}
