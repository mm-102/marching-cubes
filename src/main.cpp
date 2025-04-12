#include <cstdlib>
#include <cstdio>
#include <window_manager.hpp>
#include <shaderpregram.hpp>

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