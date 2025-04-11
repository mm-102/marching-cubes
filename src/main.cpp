#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstdio>
#include <display/window_manager.hpp>
#include <display/shaderpregram.hpp>

//Error processing callback procedure
static void error_callback(int error, const char* description) {
	fputs(description, stderr);
}

int main(){
	WindowManager windowManager(500, 500, "Marching Cubes");

	if(!windowManager.init()){
		exit(EXIT_FAILURE);
	}

	while(!windowManager.should_close()){
		windowManager.poll_events();
	}

    return EXIT_SUCCESS;
}