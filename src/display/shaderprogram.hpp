#pragma once

#include <GL/glew.h>
#include <iostream>

class ShaderProgram {
    GLuint shaderProgram;
    GLuint vertexShader;
    GLuint fragmentShader;
    static std::string readFile(std::string fileName);
    static GLuint loadShader(GLenum shaderType, const std::string &code);

public:
    ShaderProgram(const std::string &vertCode, const std::string &fragCode);
    ~ShaderProgram();
    void use();
    GLuint u(const char* varName);
    GLuint a(const char* varName);
};