// 包含上下文相关的头文件
#include "context.hpp"
#include <iostream>
// 定义全局的应用状态对象
AppState app;

/**
 * @brief 初始化应用程序所需的环境和窗口
 * 
 * 此函数主要完成以下操作：
 * 1. 初始化GLFW库
 * 2. 设置GLFW窗口的API提示为不使用客户端API
 * 3. 创建一个指定大小和标题的GLFW窗口
 */
void initApp() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    app.window = glfwCreateWindow(app.windowWidth, app.windowHeight, "Vulkan Cube Placer", nullptr, nullptr);
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

}

/**
 * @brief 清理应用程序资源
 * 
 * 此函数主要完成以下操作：
 * 1. 对应用程序中的所有统一缓冲区内存进行解映射操作
 * 2. 销毁之前创建的GLFW窗口
 * 3. 终止GLFW库的使用
 */
void cleanupApp() {
    for(const auto& it:app.uniformBuffersMemory){
        it.unmapMemory();
    }


    glfwDestroyWindow(app.window);
    glfwTerminate();
}
