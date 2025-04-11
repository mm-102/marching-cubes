#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>

//Error processing callback procedure
static void error_callback(int error, const char* description) {
	fputs(description, stderr);
}

int main(){
    GLFWwindow* window;

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
		fprintf(stderr, "Can't initialize GLFW.\n");
		exit(EXIT_FAILURE);
	}

	window = glfwCreateWindow(500, 500, "marching", NULL, NULL);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); //During vsync wait for the first refresh

    GLenum err;
	if ((err = glewInit()) != GLEW_OK) { //Initialize GLEW library
		fprintf(stderr, "Can't initialize GLEW: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}
    glfwSetTime(0);
    while (!glfwWindowShouldClose(window)){
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
	glfwTerminate();
    return EXIT_SUCCESS;
}