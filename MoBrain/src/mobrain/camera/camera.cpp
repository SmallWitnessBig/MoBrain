#include "camera.hpp"

namespace camera{
    glm::vec3 pos = glm::vec3{2.0f, 2.0f, 2.0f};
    glm::vec3 front = glm::vec3{ 0.0f, 1.0f, 0.0f };
    glm::vec3 up = glm::vec3{ 0.0f, 1.0f, 0.0f };
    float sensitivity = 0.1f;
    float yaw = -180.0f;
    float pitch = -35.0f;
    double lastX = 0.0;
    double lastY = 0.0;
    bool enableCursor = false;
    bool enableMouse = true;
    bool enableScroll = true;
    bool enableKey = true;
    float rotateSpeed = 1.0f;   
    float cameraSpeed = 5.0f;
    float fov = 45.0f;
    
    // 鼠标移动回调函数，用于处理视角旋转
    void CursorPos(){
        if (enableMouse&&app.isInGame) {
            int width, height;
            glfwGetWindowSize(app.window, &width, &height);
            double xpos, ypos;
            glfwGetCursorPos(app.window, &xpos, &ypos);
            double xoffset = xpos - width / 2.0;
            double yoffset =  height / 2.0 - ypos; // 注意这里是反过来的，因为鼠标坐标系与OpenGL不同

            yaw += xoffset * sensitivity * rotateSpeed;
            pitch += yoffset * sensitivity * rotateSpeed;

            // 限制俯仰角不超过90度，避免万向锁问题
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;

            // 根据欧拉角计算新的front向量
            front.x = std::cos(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            front.y = std::sin(glm::radians(pitch));
            front.z = std::sin(glm::radians(yaw)) * std::cos(glm::radians(pitch));
            front = glm::normalize(front);
                   

            glfwSetCursorPos(app.window, width / 2.0, height / 2.0);

        }
        

    };
            
    

    // 鼠标按钮回调函数，这里暂时保持原样，可根据需求扩展
    void MouseButton(int button, int action, int mods){

    };

    // 鼠标滚轮回调函数，用于处理相机缩放
    void Scroll(GLFWwindow* window,double xoffset, double yoffset){
        if (enableScroll&&app.isInGame) {
            if (fov >= 1.0f && fov <= 45.0f)
                fov -= yoffset*2.5f;
            if (fov <= 1.0f)
                fov = 1.0f;
            if (fov >= 45.0f)
                fov = 45.0f;
        }
    };

    // 键盘回调函数，用于处理相机移动
    void Key(){
        // 只在按键被按下或重复时更新front向量和处理移动
        if (!enableKey) return;
        
        auto deltaTime = app.frameTime / 1000.0f;
        
        if (glfwGetKey(app.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(app.window, GLFW_TRUE);
        if (app.isInGame) {
            glm::vec3 deltaPos = cameraSpeed * front * deltaTime;
            glm::vec3 deltaPos1 = glm::normalize(glm::cross(front, up)) * cameraSpeed * deltaTime;
            if (glfwGetKey(app.window, GLFW_KEY_W) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_W) == GLFW_REPEAT) {
                pos += glm::vec3{ deltaPos.x,0,deltaPos.z };
            }
            if (glfwGetKey(app.window, GLFW_KEY_S) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_S) == GLFW_REPEAT) {
                pos -= glm::vec3{ deltaPos.x,0,deltaPos.z };
            }
            if (glfwGetKey(app.window, GLFW_KEY_A) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_A) == GLFW_REPEAT) {
                pos -= glm::vec3{ deltaPos1.x,0,deltaPos1.z };
            }
            if (glfwGetKey(app.window, GLFW_KEY_D) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_D) == GLFW_REPEAT) {
                pos += glm::vec3{ deltaPos1.x,0,deltaPos1.z };
            }
            if (glfwGetKey(app.window, GLFW_KEY_SPACE) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_SPACE) == GLFW_REPEAT) {
                pos += up * cameraSpeed * deltaTime;
            }
            if (glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(app.window, GLFW_KEY_LEFT_SHIFT) == GLFW_REPEAT) {
                pos -= up * cameraSpeed * deltaTime;
            }


        }
        
    };
    void updateCamera(){
        CursorPos();
        Key();
    };
}

