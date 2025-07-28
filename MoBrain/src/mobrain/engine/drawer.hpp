#pragma once
#include "../core/context.hpp"
#include "vulkaninit.hpp"
#include "../gui/gui.hpp"
#include "camera.hpp"
#include <random>
#include <chrono>

void drawFrame();
void createSyncObjects();
void updateUniformBuffer(const uint32_t currentImage);
void framebufferResizeCallback(GLFWwindow* window, int width, int height);
