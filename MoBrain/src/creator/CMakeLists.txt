file(GLOB SOURCES 
    "*.cpp"
    "${CMAKE_SOURCE_DIR}/src/mobrain/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/mobrain/**/*.cu"
)
add_executable(MoBrain ${SOURCES}
        ../mobrain/structs/engine/graphics.hpp
        ../mobrain/structs/scene/scene.cpp
        ../mobrain/structs/gui/gui_stuct.cpp
        ../mobrain/utils/usefulMap.hpp
        ../mobrain/gui/neuronGui.cpp
        ../mobrain/gui/neuronGui.hpp
        ../mobrain/gui/usefulGui.cpp
        ../mobrain/gui/usefulGui.hpp
        ../mobrain/gui/synapseGui.cpp
        ../mobrain/gui/synapseGui.hpp
        ../mobrain/utils/ID.hpp
)
find_package(CUDAToolkit REQUIRED)
find_package(Vulkan REQUIRED)
find_package(JNI REQUIRED)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(INCLUDE 
    "${CMAKE_SOURCE_DIR}/src/mobrain"
)
target_include_directories(MoBrain PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${Vulkan_INCLUDE_DIRS}
        ${JNI_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${INCLUDE}
)

set_target_properties(MoBrain PROPERTIES
        CUDA_ARCHITECTURES 86
        CXX_STANDARD 20
        CUDA_STANDARD 20
)

target_link_libraries(MoBrain PRIVATE
        imgui
        CUDA::cudart
        Vulkan::Vulkan
        ${JNI_LIBRARIES}
        ${CMAKE_SOURCE_DIR}/lib/glfw3dll.lib
)

target_compile_options(MoBrain PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_compile_options(MoBrain PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Werror=cross-execution-space-call>
)


add_custom_command(TARGET MoBrain POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Build completed successfully"
)
