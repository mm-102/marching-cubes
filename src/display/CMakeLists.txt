file(GLOB display_src *.cpp)
add_library(libdisplay ${display_src})
target_compile_features(libdisplay PUBLIC cxx_std_20)
# set(B_PRODUCTION_MODE ON)
set_property(TARGET libdisplay PROPERTY CXX_STANDARD 20)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW 2.2 REQUIRED)

target_include_directories(libdisplay PUBLIC ${OPENGL_INCLUDE_DIR})
target_include_directories(libdisplay PUBLIC ${GLM_INCLUDE_DIRS})

target_link_libraries(libdisplay PUBLIC ${OPENGL_LIBRARIES})
target_link_libraries(libdisplay PUBLIC glfw)
target_link_libraries(libdisplay PUBLIC GLEW::GLEW)

include(FetchContent)
FetchContent_Declare(
  battery-embed
  GIT_REPOSITORY https://github.com/batterycenter/embed.git
  GIT_TAG        v1.2.19
)
FetchContent_MakeAvailable(battery-embed)

b_embed(libdisplay res/shaders/main.vert)
b_embed(libdisplay res/shaders/main.frag)