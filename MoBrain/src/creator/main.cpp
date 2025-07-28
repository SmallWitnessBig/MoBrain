// 此文件为程序的主入口文件，负责初始化应用程序、Vulkan 上下文和 ImGui 界面，
// 设置各类回调函数，处理主循环中的事件、绘制界面和计算帧率，
// 并在程序结束时进行资源清理，同时捕获并处理可能出现的异常。
#include "../mobrain/gui/gui.hpp"
#include "vulkaninit.hpp"
#include "context.hpp"
#include "camera.hpp"
#include <GLFW/glfw3.h>
#include <iostream>  
#include <chrono>

/**
 * @brief 程序主函数，负责整个应用程序的生命周期管理。
 * 
 * 1. 初始化应用程序、Vulkan 上下文和 ImGui 界面。
 * 2. 设置各类 GLFW 回调函数，用于处理用户输入和窗口事件。
 * 3. 进入主循环，处理事件、绘制界面并计算帧率。
 * 4. 程序结束时进行资源清理。
 * 5. 捕获并处理可能出现的异常。
 * 
 * @return int 程序退出状态码，0 表示正常退出，非 0 表示异常退出。
 */
void settings() {
    // 设置各类回调函数，用于处理用户输入和窗口事件
// 设置鼠标光标位置回调函数，将事件传递给 ImGui 处理
    glfwSetCursorPosCallback(app.window, [](GLFWwindow* window, double x, double y) {
        ImGui_ImplGlfw_CursorPosCallback(window, x, y);
        });
    // 设置鼠标按钮回调函数，将事件传递给 ImGui 处理
    glfwSetMouseButtonCallback(app.window, [](GLFWwindow* window, int button, int action, int mods) {
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        });

    // 设置鼠标滚动回调函数，将事件传递给 ImGui 处理
    glfwSetScrollCallback(app.window, [](GLFWwindow* window, double xoffset, double yoffset) {
        ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
        camera::Scroll(window, xoffset, yoffset);
        });
    // 设置字符输入回调函数，将事件传递给 ImGui 处理
    glfwSetCharCallback(app.window, [](GLFWwindow* window, unsigned int c) {
        ImGui_ImplGlfw_CharCallback(window, c);
        });
    // 设置键盘按键回调函数，将事件传递给 ImGui 处理
    glfwSetKeyCallback(app.window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
        });
    // 设置帧缓冲区大小回调函数，使用自定义的回调函数处理
    glfwSetFramebufferSizeCallback(app.window, framebufferResizeCallback);
    // 设置鼠标输入模式为正常模式
    glfwSetInputMode(app.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

void mainloop() {
    while (!glfwWindowShouldClose(app.window)) {
        // 记录当前帧开始时间
        auto starttime = std::chrono::high_resolution_clock::now();
        // 处理 GLFW 事件
        glfwPollEvents();

        // 更新摄像机
        camera::updateCamera();
        // 计算 FPS 和帧时间
        double currentTime = glfwGetTime();
        app.frameCount++;
        // 每秒更新一次 FPS 和帧时间
        if (currentTime - app.lastTime >= 1.0) {
            app.fps = app.frameCount;
            app.frameTime = 1000.0 / app.fps;
            app.frameCount = 0;
            app.lastTime = currentTime;
        }

        // 绘制 ImGui 界面
        drawImGui();

        // 绘制一帧画面
        drawFrame();

        // 计算当前帧的耗时
        auto endTime = std::chrono::high_resolution_clock::now();
        auto frameTime = std::chrono::duration<float, std::chrono::seconds::period>(endTime - starttime).count();
        app.frameTime = frameTime * 1000.0f;
    }
}
void createScene() {
    for (float i = 0; i <50; i+=1.0f)
    {
        for (float j = 0; j <50; j+=1.0f)
        {
            auto c = std::make_shared<Cube>(glm::vec3{ i,j,0.0f }, glm::vec3{ i/10,j/10,0.0f });
            app.render_scene.addObject(c);
            app.bufferM.addObject(c);
        }
    }
    
}

int main() {
    try {

        // 初始化阶段
        // initApp() 函数用于初始化 GLFW 窗口
        initApp();      //glfw window
        // initVulkan() 函数用于初始化 Vulkan 上下文
		initVulkan();   //vulkan context
        // initImGui() 函数用于初始化 ImGui 上下文
        initImGui();    //ImGui context
        std::cout << "init finished" << std::endl;
        createScene();

        settings();

        mainloop();

        // 等待 Vulkan 设备完成所有操作
        app.device.waitIdle();

        // 清理 ImGui 资源
        cleanupImGui();
        // 清理应用程序资源
        cleanupApp();
    }
    // 捕获 vk::SystemError 异常并输出错误信息
    catch (const vk::SystemError& err) {
        // use err.code() to check err type
        std::cerr<<"vk::SystemError - code: {} "<<err.code().message();
        std::cerr<<"vk::SystemError - what: {}"<<err.what();
    }
    // 捕获标准异常并输出错误信息，程序异常退出
    catch(const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
	}
    return 0;
}
