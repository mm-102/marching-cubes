#include "shaderpregram.hpp"

#include <sstream>
#include <fstream>

std::string ShaderProgram::readFile(std::string fileName){
    std::stringstream buf;
    std::ifstream file(fileName);
	if(!file.is_open()){
		std::cerr << "Error opening file: " << fileName << std::endl;
		return "";
	}
    buf << file.rdbuf();
    return buf.str();
}

GLuint ShaderProgram::loadShader(GLenum shaderType, const std::string &code) {
    GLuint shader = glCreateShader(shaderType);
	const GLchar* s[] = { code.c_str() };

    glShaderSource(shader, 1, s, NULL);
    glCompileShader(shader);

    int infologLength = 0;
	int charsWritten = 0;
	char* infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 1) {
		infoLog = new char[infologLength];
		glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
		std::cerr << "Shader compile info: " << infoLog << std::endl;
		delete[] infoLog;
	}

	return shader;
}

ShaderProgram::ShaderProgram(const std::string &vertCode, const std::string &fragCode){
    this->vertexShader = ShaderProgram::loadShader(GL_VERTEX_SHADER, vertCode);
    this->fragmentShader = ShaderProgram::loadShader(GL_FRAGMENT_SHADER, fragCode);
    this->shaderProgram = glCreateProgram();

    int infologLength = 0;
	int charsWritten = 0;
	char* infoLog;

	glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 1)
	{
		infoLog = new char[infologLength];
		glGetProgramInfoLog(shaderProgram, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		delete[]infoLog;
	}

}

ShaderProgram::~ShaderProgram(){
	glDetachShader(shaderProgram, vertexShader);
	glDetachShader(shaderProgram, fragmentShader);

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	glDeleteProgram(shaderProgram);
}

//Make the shader program active
void ShaderProgram::use() {
	glUseProgram(shaderProgram);
}

//Get the slot number corresponding to the uniform variableName
GLuint ShaderProgram::u(const char* variableName) {
	return glGetUniformLocation(shaderProgram, variableName);
}

//Get the slot number corresponding to the attribute variableName
GLuint ShaderProgram::a(const char* variableName) {
	return glGetAttribLocation(shaderProgram, variableName);
}