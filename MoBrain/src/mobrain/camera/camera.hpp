#pragma once
#include "core/context.hpp"
#include <functional>

namespace camera{
    extern glm::vec3 pos;
    extern glm::vec3 front;
    extern glm::vec3 up;
    extern float sensitivity;
    extern float yaw;
    extern float pitch;
    
    extern double lastX;
    extern double lastY;
    extern bool enableCursor;
    extern bool enableMouse;
    extern bool enableScroll;
    extern bool enableKey;
    extern float cameraSpeed;
    extern float rotateSpeed;
    extern float fov;

    void CursorPos();
    void MouseButton(int button, int action, int mods);
    void Scroll(GLFWwindow* window,double xoffset, double yoffset);
    void Key();
    void updateCamera();
}

