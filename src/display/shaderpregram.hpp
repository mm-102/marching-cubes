#pragma once

#include <GL/glew.h>
#include <iostream>

class ShaderProgram {
    GLuint shaderProgram;
    GLuint vertexShader;
    GLuint fragmentShader;
    std::string readFile(std::string fileName);
    GLuint loadShader(GLenum shaderType, const std::string fileName);

public:
    ShaderProgram(const std::string vertFile, const std::string fragFile);
    ~ShaderProgram();
    void use();
    GLuint u(const char* varName);
    GLuint a(const char* varName);
};