#version 400

out vec4 fragColor;

in vec3 vPos;
in vec3 vNorm;

uniform vec3 uCamera;
uniform vec4 uModulate = vec4(1.0);

uniform vec4 uInModulate = vec4(1.0);

uniform vec3 uAmbientLight = vec3(0.1);
uniform vec3 uDiffuseLight = vec3(0.9);
uniform vec3 uSpecularLight = vec3(1.0);

void main(){
    vec3 viewDir = normalize(uCamera-vPos);
    vec3 lightDir = normalize(uCamera); // light from camera to (0,0,0)

    float d = dot(vNorm, lightDir);
    float diff = abs(d);

    vec3 reflectDir = reflect(-lightDir, vNorm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    // vec3 res = uModulate.rgb * (uAmbientLight + uDiffuseLight * diff + uSpecularLight * spec);
    vec3 res = mix(uInModulate.rgb, uModulate.rgb, d * 0.5 + 0.5) * (uAmbientLight + uDiffuseLight * diff + uSpecularLight * spec);


    fragColor = vec4(res, 1.0);
}
