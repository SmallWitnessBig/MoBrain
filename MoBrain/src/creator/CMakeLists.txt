file(GLOB SOURCES 
    "*.cpp"
    "${CMAKE_SOURCE_DIR}/src/mobrain/**/*.cpp"
)
add_executable(MoBrain ${SOURCES})

set(INCLUDE 
    "${CMAKE_SOURCE_DIR}/src/mobrain/camera"
    "${CMAKE_SOURCE_DIR}/src/mobrain/core"
    "${CMAKE_SOURCE_DIR}/src/mobrain/engine"
    "${CMAKE_SOURCE_DIR}/src/mobrain/gui"
    "${CMAKE_SOURCE_DIR}/src/mobrain/utils"
)
target_include_directories(MoBrain PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${INCLUDE}
)

target_link_libraries(MoBrain PRIVATE
    imgui  
    ${CMAKE_SOURCE_DIR}/lib/glfw3dll.lib
    ${CMAKE_SOURCE_DIR}/lib/vulkan-1.lib
    ${CMAKE_SOURCE_DIR}/lib/opencv_world4120.lib
    ${CMAKE_SOURCE_DIR}/lib/opencv_world4120d.lib
)

