#version 400

layout (location=0) in vec3 aPos;
layout (location=1) in vec3 aNorm;

out vec3 vPos;
out vec3 vNorm;

uniform mat4 P;
uniform mat4 V;
uniform mat4 M;

void main(){
    vPos = vec3(M * vec4(aPos,1.0));
    vNorm = mat3(transpose(inverse(M))) * aNorm;
    if(length(vNorm)>0)
		vNorm = normalize(vNorm);

    gl_Position = P * V * M * vec4(aPos, 1.0);
}
