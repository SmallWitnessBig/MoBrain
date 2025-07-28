add_library(imgui STATIC
    imgui.cpp
    imgui_demo.cpp
    imgui_draw.cpp
    imgui_tables.cpp
    imgui_widgets.cpp
    backends/imgui_impl_glfw.cpp
    backends/imgui_impl_vulkan.cpp
)

target_include_directories(imgui PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/backends
)

target_include_directories(imgui PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# �����Ҫ�ĺ�
target_compile_definitions(imgui PRIVATE 
    IMGUI_IMPL_VULKAN_HAS_DYNAMIC_RENDERING
)