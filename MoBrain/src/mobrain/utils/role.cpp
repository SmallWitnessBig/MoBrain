#include "camera/camera.hpp"
#include "core/context.hpp"

Role::Role() {
    focusCube = nullptr;
    focusNeuronGroup = nullptr;
    place = new Cube(glm::vec3{ 0,0,0 }, glm::vec4{ 1,1,1,1 });
}
void Role::Key() {
    if (app.isInGame) {
        double currenTime = glfwGetTime();
        if (glfwGetKey(app.window, GLFW_KEY_C) == GLFW_PRESS) {
            if (currenTime - lastCClickTime > clickCooldown) {
                app.guiFlags.isOpenNetGui = !app.guiFlags.isOpenNetGui;
                printf("isOpenNetGui :%.hhd", app.guiFlags.isOpenNetGui);
                lastCClickTime = currenTime;
            }
        }
    }
}

void Role::MouseButton() {
    if (app.isInGame) {
        auto& s = app.render_scene;
        auto& n = app.net;
        double currentTime = glfwGetTime();
        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (currentTime - lastLeftClickTime > clickCooldown) {
                if (app.guiFlags.isOpenNetGui) {
                    app.guiFlags.isOpenNetGui=false;
                }
                s.removeCube(focusCube);

                focusCube = nullptr; // 移除后重置焦点
                lastLeftClickTime = currentTime;
            }
        }
        if (glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            if (!s.pos_map.contains(place->pos)
                && app.isFocus
                && currentTime - lastRightClickTime > clickCooldown) {

                const Cube* p = new Cube(place->pos, place->color);
                s.addCube(p);
                lastRightClickTime = currentTime;

            }
        }

    }
}
glm::vec3 findFace(const Cube* focus) {
    if (!focus) return glm::vec3(0);

    const glm::vec3 minBound = focus->pos - 0.5f;
    const glm::vec3 maxBound = focus->pos + 0.5f;
    const glm::vec3 rayStart = camera::pos;
    const glm::vec3 rayDir = glm::normalize(camera::front);

    float tMin = (minBound.x - rayStart.x) / rayDir.x;
    float tMax = (maxBound.x - rayStart.x) / rayDir.x;
    if (tMin > tMax) std::swap(tMin, tMax);

    float tyMin = (minBound.y - rayStart.y) / rayDir.y;
    float tyMax = (maxBound.y - rayStart.y) / rayDir.y;
    if (tyMin > tyMax) std::swap(tyMin, tyMax);

    if ((tMin > tyMax) || (tyMin > tMax))
        return focus->pos; // 无交点

    tMin = std::max(tMin, tyMin);
    tMax = std::min(tMax, tyMax);

    float tzMin = (minBound.z - rayStart.z) / rayDir.z;
    float tzMax = (maxBound.z - rayStart.z) / rayDir.z;
    if (tzMin > tzMax) std::swap(tzMin, tzMax);

    if ((tMin > tzMax) || (tzMin > tMax))
        return focus->pos; // 无交点

    tMin = std::max(tMin, tzMin);

    // 只考虑正方向的交点
    if (tMin < 0) return focus->pos;

    // 计算交点位置
    const glm::vec3 hitPoint = rayStart + tMin * rayDir;

    // 确定相交的面
    constexpr float epsilon = 0.001f;
    glm::vec3 faceOffset(0.0f);

    if (std::abs(hitPoint.x - minBound.x) < epsilon) faceOffset.x = -1.0f;
    else if (std::abs(hitPoint.x - maxBound.x) < epsilon) faceOffset.x = 1.0f;
    else if (std::abs(hitPoint.y - minBound.y) < epsilon) faceOffset.y = -1.0f;
    else if (std::abs(hitPoint.y - maxBound.y) < epsilon) faceOffset.y = 1.0f;
    else if (std::abs(hitPoint.z - minBound.z) < epsilon) faceOffset.z = -1.0f;
    else if (std::abs(hitPoint.z - maxBound.z) < epsilon) faceOffset.z = 1.0f;

    return focus->pos + faceOffset;
}
Role& Role::update() {
    auto& scene = app.render_scene;
    focusCube = nullptr;
    place->color = glm::vec4{ 1.0f, 0.0f, 0.0f, 1.0f };

    const float maxDistance = 15.0f;
    const float step = 0.1f;
    float distance = 0.0f;

    while (distance < maxDistance) {
        glm::vec3 target = camera::pos + distance * camera::front;

        glm::vec3 intTarget = {
            std::floor(target.x + 0.5f),
            std::floor(target.y + 0.5f),
            std::floor(target.z + 0.5f)
        };

        auto it = scene.pos_map.find(intTarget);
        if (it != scene.pos_map.end() && it->second < scene.cubes.size()) {
            focusCube = scene.cubes[it->second];
            place->pos = findFace(focusCube);
            place->model = glm::translate(glm::mat4(1.0f), place->pos);
            break;
        }

        distance += step;
    }

    if (!focusCube) {
        app.isFocus = false;
    }
    else {
        app.isFocus = true;
    }
    MouseButton();
    Key();
    return *this;
}