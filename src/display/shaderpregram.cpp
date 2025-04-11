#include <display/shaderpregram.hpp>

#include <sstream>
#include <fstream>

std::string ShaderProgram::readFile(std::string fileName){
    std::stringstream buf;
    std::ifstream file(fileName);
    buf << file.rdbuf();
    return buf.str();
}

GLuint ShaderProgram::loadShader(GLenum shaderType, std::string fileName) {
    GLuint shader = glCreateShader(shaderType);

    std::string shaderSource = readFile(fileName);
	const GLchar* s[] = { shaderSource.c_str() };

    glShaderSource(shader, 1, s, NULL);
    glCompileShader(shader);

    int infologLength = 0;
	int charsWritten = 0;
	char* infoLog;

	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 1) {
		infoLog = new char[infologLength];
		glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
		std::cout << infoLog << std::endl;
		delete[] infoLog;
	}

	//Return shader handle
	return shader;
}

ShaderProgram::ShaderProgram(const std::string vertFile, const std::string fragFile){
    this->vertexShader = loadShader(GL_VERTEX_SHADER, vertFile);
    this->fragmentShader = loadShader(GL_FRAGMENT_SHADER, vertFile);
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